import mujoco
import mujoco.viewer
import numpy as np
from deploy.deploy_mujoco.utils.math import euler_from_quat, quat_rotate_inverse
from legged_gym import LEGGED_GYM_ROOT_DIR
from deploy.deploy_mujoco.configs import Go2wPiperCfg
import torch

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

        # joint states
        self.joint_pos = np.zeros(self.joint_num)
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
        # scales
        self.obs_scales = self.cfg.normalization.obs_scales
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.commands_scales = np.array([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel])
        # ee goal
        self.ee_goal_pos = np.array([0.4, 0.0, 0.3])
        self.ee_goal_orn = np.array([0.5, -0.5, -0.5, -0.5])
        self.ee_goal = np.concatenate([self.ee_goal_pos, self.ee_goal_orn])
        # counter
        self.counter = 0.0
        self.decimation = self.cfg.control.decimation

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

    def get_joint_states(self):
        for j in range(self.joint_num):
            # joint pos
            qpos_adr = self.model.jnt_qposadr[j + 1]
            self.joint_pos[j] = self.data.qpos[qpos_adr]

            # joint vel
            dof_adr = self.model.jnt_dofadr[j + 1]
            self.joint_vel[j] = self.data.qvel[dof_adr]    

        self.joint_pos[self.wheel_indices] = 0

    def get_body_states(self):
        self.base_pos = self.data.qpos[0:3]
        self.base_quat = self.data.qpos[3:7]
        self.base_vel = quat_rotate_inverse(self.base_quat, self.data.qvel[0:3])
        self.base_angle_vel = quat_rotate_inverse(self.base_quat, self.data.qvel[3:6])

    def compute_torques(self, actions):
        # pos ref
        actions_scaled = actions * self.motor_strength * self.cfg.control.action_scale 
        actions_scaled[self.wheel_indices] = 0 

        # vel ref
        vel_ref = np.zeros(self.joint_num)
        vel_tmp = actions * self.motor_strength * self.cfg.control.action_scale_vel
        vel_ref[self.wheel_indices] = vel_tmp[self.wheel_indices]

        torques = self.p_gains * (actions_scaled - self.joint_pos) + self.d_gains * (vel_ref - self.joint_vel)
        return torques
    
    def compute_observations(self):
        self.obs_buf = np.concatenate([
                                        self.get_body_orientation(),
                                        self.base_angle_vel * self.obs_scales.ang_vel,
                                        self.joint_pos * self.obs_scales.dof_pos,
                                        self.joint_vel * self.obs_scales.dof_vel,
                                        self.actions[:self.num_leg_actions],
                                        self.commands[:3] * self.commands_scales,
                                        self.ee_goal_pos
        ])

        self.obs_buf = np.clip(self.obs_buf, -self.clip_obs, self.clip_obs)

        self.obs_history_buf = np.concatenate([self.obs_history_buf[1:, :], self.obs_buf[None, :]], axis=0)
    
    def get_body_orientation(self):
        r, p, y = euler_from_quat(self.base_quat)
        body_angles = np.array([r, p, y])
        return body_angles[:-1]
    
    def step(self):

        mujoco.mj_step(self.model, self.data)

        # get obs
        self.get_joint_states()
        self.get_body_states()
        self.compute_observations()
        self.counter += 1

        if self.counter % self.decimation == 0:
            # load policy
            obs_history_buf = torch.from_numpy(self.obs_history_buf.reshape(-1)).unsqueeze(0).float()
            latent = self.hist_encoder_policy(obs_history_buf)
            obs_buf = torch.from_numpy(self.obs_buf).unsqueeze(0).float()
            obs_tensor = torch.cat([obs_buf, latent], dim=1)
            actions = self.actor_policy(obs_tensor).detach().cpu().numpy().squeeze()

            # actions clip
            actions[self.num_leg_actions:] = 0.0
            self.actions = np.clip(actions, -self.clip_actions, self.clip_actions)

        # compute torques
        self.torques= self.compute_torques(self.actions)

def main():
    robot = Go2wPiper(cfg = Go2wPiperCfg)
    robot._init_robot()
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            robot.step()
            viewer.sync() 

if __name__ == "__main__":
    main()