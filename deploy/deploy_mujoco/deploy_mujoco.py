import mujoco
import mujoco.viewer
import numpy as np
from deploy.deploy_mujoco.utils.math import euler_from_quat, quat_apply, wrap_to_pi, quat_rotate_inverse, quat_relative
from deploy.deploy_mujoco.utils.logger import EETrackingLogger
from deploy.deploy_mujoco.modules.ee_tracking import DLSIKController, EEGoalSampler, EEPointVisualizer
from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_mujoco.configs import Go2wPiperCfg
import torch
import time

class Go2wPiper:
    def __init__(self, cfg: Go2wPiperCfg):
        self.cfg = cfg
        xml_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        actor_path = self.cfg.asset.actor_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        hist_encoder_path = self.cfg.asset.hist_encoder_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.actor_policy = torch.jit.load(actor_path)
        self.hist_encoder_policy = torch.jit.load(hist_encoder_path)

    def _init_robot(self):
        # joint names
        self.joint_names = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if isinstance(name, str):
                self.joint_names.append(name)
        self.joint_num = len(self.joint_names)

        # wheel indices
        self.wheel_indices = []
        for wheel_name in self.cfg.asset.wheel_names:
            jid = self.joint_names.index(wheel_name)
            self.wheel_indices.append(jid)
        
        # PD parameters
        self.p_gains = np.zeros(self.joint_num)
        self.d_gains = np.zeros(self.joint_num)
        for i in range(self.joint_num):
            name = self.joint_names[i]
            for joint_name in self.cfg.control.stiffness.keys():
                if joint_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[joint_name]
                    self.d_gains[i] = self.cfg.control.damping[joint_name]

        # init state
        self.default_joint_pos = np.zeros(self.joint_num)
        for i in range(self.joint_num):
            name = self.joint_names[i]
            for joint_name in self.cfg.init_state.default_joint_angles.keys():
                if joint_name in name:
                    self.default_joint_pos[i] = self.cfg.init_state.default_joint_angles[joint_name]

        # joint states
        self.joint_pos = np.zeros(self.joint_num)
        self.joint_err = np.zeros(self.joint_num)
        self.joint_vel = np.zeros(self.joint_num)
        # obs
        self.num_proprio = self.cfg.env.num_proprio
        self.history_len = self.cfg.env.history_len
        self.obs_history_buf = np.zeros((self.history_len, self.num_proprio))
        # actions
        self.actions = np.zeros(self.cfg.env.num_actions)
        self.torques = np.zeros(self.cfg.env.num_actions)
        self.num_leg_actions = self.cfg.env.num_leg_actions
        self.num_arm_actions = self.cfg.env.num_arm_actions
        # commands
        self.commands = np.zeros(self.cfg.commands.num_commands)
        self.commands_ranges = self.cfg.commands.ranges
        self.heading_command = self.cfg.commands.heading_command
        # scales
        self.obs_scales = self.cfg.normalization.obs_scales
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.commands_scales = np.array([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel])
        # arm info
        self.ee_goal_pos_final = np.zeros(3)
        self.ee_goal_pos = np.zeros(3)
        self.ee_goal_orn = np.array([1.0, 0.0, 0.0, 0.0]) # wxyz
        self.curr_ee_goal_pos_world = np.zeros(3)
        self.step_sample_time = 0.0
        self.arm_target_angles = np.zeros(self.num_arm_actions)
        # 
        self.decimation = self.cfg.control.decimation
        self.forward_vec = [1., 0., 0.]

        # ik controller
        self.ik_controller = DLSIKController(self.model, self.data, self.num_arm_actions, 
                                             site_name="ee_site", damping=0.05, step_size=1.0)
        # ee goal sampler
        self.ee_goal_sampler = EEGoalSampler(self.cfg.goal_ee)

        # ee point visualizer
        self.ee_point_vis = EEPointVisualizer(
            enabled_points=self.cfg.goal_ee.vis_enabled_points,
            point_size=self.cfg.goal_ee.vis_point_size,
        )

        # domain rand
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = np.concatenate([
                np.random.uniform(self.cfg.domain_rand.leg_motor_strength_range[0], 
                                  self.cfg.domain_rand.leg_motor_strength_range[1],
                                  size=self.num_leg_actions),
                np.random.uniform(self.cfg.domain_rand.arm_motor_strength_range[0], 
                                  self.cfg.domain_rand.arm_motor_strength_range[1],
                                  size=self.num_arm_actions)  
            ])
        else:
            self.motor_strength = np.ones(self.cfg.env.num_actions)

    def get_sensor_data(self, name):
        id_ = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = self.model.sensor_adr[id_], self.model.sensor_dim[id_]
        return self.data.sensordata[adr: adr + dim]

    def get_joint_states(self):
        for i, n in enumerate(self.joint_names):
            self.joint_pos[i] = self.get_sensor_data(n + "_pos")[0]
            self.joint_vel[i] = self.get_sensor_data(n + "_vel")[0]
        self.joint_pos[self.wheel_indices] = 0
        self.joint_err = self.joint_pos - self.default_joint_pos
        self.joint_err[self.wheel_indices] = 0

    def get_body_states(self):
        # base states
        self.base_quat = self.get_sensor_data("base_quat")
        self.base_euler = self.get_base_euler(self.base_quat)
        self.base_angle_vel = self.get_sensor_data("base_gyro")
        # arm base states
        self.arm_base_pos = self.get_sensor_data("arm_base_pos")
        self.arm_base_quat = self.get_sensor_data("arm_base_quat")
        # ee states
        self.ee_pos = self.get_sensor_data("ee_pos")
        self.ee_quat = self.get_sensor_data("ee_quat")

    def compute_torques(self, actions):
        # pos ref
        actions_scaled = actions * self.motor_strength * self.cfg.control.action_scale 
        actions_scaled[self.wheel_indices] = 0 
        pos_ref = actions_scaled + self.default_joint_pos
        # vel ref
        vel_ref = np.zeros(self.joint_num)
        vel_tmp = actions * self.motor_strength * self.cfg.control.action_scale_vel
        vel_ref[self.wheel_indices] = vel_tmp[self.wheel_indices]

        torques = self.p_gains * (pos_ref - self.joint_pos) + self.d_gains * (vel_ref - self.joint_vel)
        return torques
    
    def compute_observations(self):

        self.obs_buf = np.concatenate([
                                        self.base_euler[:2],
                                        self.base_angle_vel * self.obs_scales.ang_vel,
                                        self.joint_err * self.obs_scales.dof_pos,
                                        self.joint_vel * self.obs_scales.dof_vel,
                                        self.actions[:self.num_leg_actions],
                                        self.commands[:3] * self.commands_scales,
                                        self.ee_goal_pos
        ])
        self.obs_buf = np.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        self.obs_history_buf = np.concatenate([self.obs_history_buf[1:, :], self.obs_buf[None, :]], axis=0)
    
    def get_base_euler(self, base_quat):
        r, p, y = euler_from_quat(base_quat)
        base_euler = np.array([r, p, y], dtype=np.float32)
        return base_euler
    
    def set_commands(self, cmds):
        self.commands = cmds
        self.commands[0] = np.clip(self.commands[0], self.commands_ranges.lin_vel_x[0], self.commands_ranges.lin_vel_x[1])
        self.commands[1] = np.clip(self.commands[1], self.commands_ranges.lin_vel_y[0], self.commands_ranges.lin_vel_y[1])
        self.commands[2] = np.clip(self.commands[2], self.commands_ranges.ang_vel_yaw[0], self.commands_ranges.ang_vel_yaw[1])
        self.commands[3] = np.clip(self.commands[3], self.commands_ranges.heading[0], self.commands_ranges.heading[1])
        if self.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = np.arctan2(forward[1], forward[0])
            self.commands[2] = np.clip(2.0 * wrap_to_pi(self.commands[3] - heading), 
                                       self.commands_ranges.ang_vel_yaw[0], self.commands_ranges.ang_vel_yaw[1])

    def step(self):
        # get states
        self.get_joint_states()
        self.get_body_states()
        self.step_sample_time = self.data.time
        # set cmds
        cmds = [0.5, 0.0, 0.0, 0.0]
        self.set_commands(cmds)
        # ee goal update
        self.ee_goal_sampler._update_curr_goal()
        self.ee_goal_pos_final = self.ee_goal_sampler.ee_goal_cart
        self.ee_goal_pos = self.ee_goal_sampler.curr_ee_goal_cart
        self.ee_goal_orn = self.ee_goal_sampler.ee_goal_orn_quat
        # get obs
        self.compute_observations()

        # load policy
        obs_history_buf = torch.from_numpy(self.obs_history_buf.reshape(-1)).unsqueeze(0).float()
        latent = self.hist_encoder_policy(obs_history_buf)
        obs_buf = torch.from_numpy(self.obs_buf).unsqueeze(0).float()
        obs_tensor = torch.cat([obs_buf, latent], dim=1)
        actions = self.actor_policy(obs_tensor).detach().cpu().numpy().squeeze()
        # actions clip
        self.actions = np.clip(actions, -self.clip_actions, self.clip_actions)

        # ik solver
        ee_pos_local = quat_rotate_inverse(self.arm_base_quat, self.ee_pos - self.arm_base_pos)
        ee_orn_local = quat_relative(self.arm_base_quat, self.ee_quat)
        self.arm_target_angles = self.ik_controller.solve(self.ee_goal_pos, 
                                                          self.ee_goal_orn, 
                                                          ee_pos_local,
                                                          ee_orn_local, 
                                                          self.joint_pos[-self.num_arm_actions:])

        # ctrl
        self.torques = self.compute_torques(self.actions)
        self.data.ctrl[:] = self.torques[:self.num_leg_actions]
        self.data.qpos[-self.num_arm_actions:] = self.arm_target_angles

        # visualize trajectories
        ee_start_pos_world = quat_apply(self.arm_base_quat, self.ee_goal_sampler.ee_start_cart) + self.arm_base_pos
        self.curr_ee_goal_pos_world = quat_apply(self.arm_base_quat, self.ee_goal_pos) + self.arm_base_pos
        ee_goal_pos_world = quat_apply(self.arm_base_quat, self.ee_goal_pos_final) + self.arm_base_pos
        self.ee_point_vis.set_point("start", ee_start_pos_world)
        self.ee_point_vis.set_point("actual", self.ee_pos)
        self.ee_point_vis.set_point("target", ee_goal_pos_world)

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

def main():
    # init
    robot = Go2wPiper(cfg = Go2wPiperCfg)
    robot._init_robot()
    ee_logger = EETrackingLogger(start_time_s=1.0, end_time_s=10.0)
    # running
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            robot.step()
            ee_logger.log_sample(
                robot.step_sample_time,
                robot.ee_pos,
                robot.curr_ee_goal_pos_world,
            )
            ee_logger.maybe_print(robot.data.time)
            robot.ee_point_vis.render(viewer)
            viewer.sync()
            time_until_next_step = robot.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
