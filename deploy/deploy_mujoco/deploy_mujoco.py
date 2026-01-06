import mujoco
import mujoco.viewer
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from arm_controllers import Robot, CartesianIKController
from deploy.deploy_mujoco.configs import Go2wPiperCfg

class Go2wPiper:
    def __init__(self, cfg: Go2wPiperCfg):
        self.cfg = cfg
        self.xml_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

    def _init_robot(self):
        # arm
        self.arm = Robot(dof = self.cfg.env.num_arm_actions, 
                  base_link = self.cfg.asset.arm_base_name, 
                  end_link = self.cfg.asset.gripper_name, 
                  joints = self.cfg.asset.arm_joint_names, 
                  actuators = self.cfg.asset.arm_actuator_names)
        self.arm_controller = CartesianIKController(self.arm)
        self.arm_controller.set_model(self.model)
        self.arm_controller.set_data(self.data)
        
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
        
    def get_joint_states(self):
        for j in range(self.joint_num):
            # joint pos
            qpos_adr = self.model.jnt_qposadr[j + 1]
            self.joint_pos[j] = self.data.qpos[qpos_adr]

            # joint vel
            dof_adr = self.model.jnt_dofadr[j + 1]
            self.joint_vel[j] = self.data.qvel[dof_adr]    

def main():
    robot = Go2wPiper(cfg = Go2wPiperCfg)
    robot._init_robot()

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(robot.model, robot.data)
            viewer.sync() 

if __name__ == "__main__":
    main()