class Go2wPiperCfg:
    class env:
        num_leg_actions = 16
        num_arm_actions = 6
        num_actions = num_leg_actions + num_arm_actions
        num_proprio = 2 + 3 + 22 + 22 + 16 + 3 + 3
        history_len = 10

    class commands:
        num_commands = 4
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1, 1] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/robots/go2w_piper/scene.xml'
        actor_path = '{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2w_piper/traced_actor.pt'
        hist_encoder_path = '{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2w_piper/traced_hist_encoder.pt'
        wheel_names = ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]
        arm_base_name = "arm_base"
        gripper_name = "link7"
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        arm_actuator_names = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6"]

    class control:
        # PD Drive parameters:
        stiffness = {'hip_joint': 40.,'thigh_joint': 40.,'calf_joint': 40.,"wheel_joint": 0.}  # [N*m/rad]
        damping = {'hip_joint': 1,'thigh_joint': 1,'calf_joint': 1,"wheel_joint": 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        action_scale_vel = 10.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class init_state:
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.67,   # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'FR_thigh_joint': 0.67,   # [rad]
            'RR_thigh_joint': 0.67,   # [rad]

            'FL_calf_joint': -1.3,    # [rad]
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -1.3,    # [rad]
            'RR_calf_joint': -1.3,    # [rad]
            
            'FL_wheel_joint': 0.0,
            'RL_wheel_joint': 0.0,
            'FR_wheel_joint': 0.0,
            'RR_wheel_joint': 0.0,

            # arm joint
            'joint1': 0.0,
            'joint2': 0.0,
            'joint3': 0.0,
            'joint4': 0.0,
            'joint5': 0.0,
            'joint6': 0.0,
        }

    class domain_rand:
        randomize_motor = False
        leg_motor_strength_range = [0.9, 1.1]
        arm_motor_strength_range = [0.9, 1.1]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 100.