class Go2wPiperCfg:
    class env:
        num_leg_actions = 16
        num_arm_actions = 6
        num_actions = num_leg_actions + num_arm_actions
        num_proprio = 2 + 3 + 22 + 22 + 16 + 3 + 3
        num_priv = 5 + 1 + 16
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/robots/go2w_piper/scene.xml'
        wheel_names = ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]
        arm_base_name = "arm_base"
        gripper_name = "link7"
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        arm_actuator_names = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6"]

    class control:
        # PD Drive parameters:
        stiffness = {'hip_joint': 40.,'thigh_joint': 40.,'calf_joint': 40.,"foot_joint": 0.}  # [N*m/rad]
        damping = {'hip_joint': 1,'thigh_joint': 1,'calf_joint': 1,"foot_joint": 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        action_scale_vel = 10.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5