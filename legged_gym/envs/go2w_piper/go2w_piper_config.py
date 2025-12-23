from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Go2wPiperCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 
        num_leg_actions = 16
        num_arm_actions = 6
        num_actions = num_leg_actions + num_arm_actions
        num_proprio = 2 + 3 + 22 + 22 + 16 + 3 + 3
        num_priv = 5 + 1 + 16
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv
        num_privileged_obs = None

    class goal_ee:
        arm_base_offset = [0, 0, 0.06]
        traj_time = [1, 3]
        hold_time = [1, 2]
        collision_upper_limits = [0.35, 0.25, -0.05]
        collision_lower_limits = [-0.35, -0.25, -0.6]
        underground_limit = -0.6
        num_collision_check_samples = 10
        max_resample_attempts = 10

        class sphere_center:
            x_offset = 0 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.6 # Relative to terrain

        class ranges:
            init_pos_start = [0.5, 0.3, 0]
            init_pos_end = [0.5, 0.6, 0]
            pos_l = [0.4, 0.7]
            pos_p = [0, np.pi/3]
            pos_y = [-1.2, 1.2]
            
            default_ee_rpy = [0, -np.pi/2, -np.pi/2]
            delta_orn_r = [-0.1, 0.1]
            delta_orn_p = [-0.1, 0.1]
            delta_orn_y = [-0.1, 0.1]

    class commands( LeggedRobotCfg ):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1, 1] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
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
            
            'FL_foot_joint': 0.0,
            'RL_foot_joint': 0.0,
            'FR_foot_joint': 0.0,
            'RR_foot_joint': 0.0,

            # arm joint
            'joint1': 0.0,
            'joint2': 1.57,
            'joint3': -0.8,
            'joint4': 0.0,
            'joint5': -0.7,
            'joint6': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_joint': 40.,'thigh_joint': 40.,'calf_joint': 40.,"foot_joint":0}  # [N*m/rad]
        damping = {'hip_joint': 1,'thigh_joint': 1,'calf_joint': 1,"foot_joint":0.5}     # [N*m*s/rad]
        arm_joint_stiffness = 400
        arm_joint_damping = 40
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        action_scale_vel = 10.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w_piper_description/urdf/go2w_piper_description.urdf'
        name = "go2w_piper_description"
        foot_name = "foot"
        wheel_name = "foot"
        gripper_name = "link7"
        mirror_joint_name = [
            ["FL_(hip|thigh|calf).*", "FR_(hip|thigh|calf).*"],
            ["RL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter "base","calf","hip","thigh"
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        fix_base_link = False

    class domain_rand:
        randomize_friction = False
        friction_range = [0.25, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 0.3]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.
        randomize_base_com = False
        added_com_range_x = [-0.05, 0.05]
        added_com_range_y = [-0.05, 0.05]
        added_com_range_z = [-0.05, 0.05]
        randomize_gripper_mass = False
        gripper_added_mass_range = [0, 0.1]
        randomize_motor = False
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]

    class rewards( LeggedRobotCfg.rewards ):
        reward_container_name = "go2w_piper_rewards"

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1.0
        soft_dof_pos_limit = 1.0 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.4
        max_contact_force = 100. # forces above this value are penalized
       
        class leg_scales:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -0.1
            ang_vel_xy = -0.1
            orientation = -1.0
            torques = -0.0005
            dof_vel = -1e-7
            dof_acc = -1e-7
            base_height = -0.5
            feet_air_time = 0.0
            collision = -0.1
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.2
            dof_pos_limits = -0.5
            run_still = -0.1
            joint_power = -1e-5
            joint_mirror = -0.5

        class arm_scales:
            termination = -0.0
            tracking_ee_cart = 0.0
            tracking_ee_cart_world = 1.0
            tracking_ee_orn = 0.5

class Go2wPiperCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_std = [[1.0, 1.0, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_tanh = False
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]
        priv_encoder_dims = [64, 20]
        num_leg_actions = Go2wPiperCfg.env.num_leg_actions
        num_arm_actions = Go2wPiperCfg.env.num_arm_actions

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 2e-4 
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.2, 0.2, 0.2, 0.2] * 4 + [0.2] * 6]
        mixing_schedule = [1.0, 0, 3000]
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48
        max_iterations = 15000 # number of policy updates
        save_interval = 200
        run_name = ''
        experiment_name = 'go2w_piper'
        load_run = -1
        checkpoint = -1