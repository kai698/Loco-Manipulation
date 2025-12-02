from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 5)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.25, 1.25]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.restitution_range = [0.0, 0.3]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [-1., 1.]
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.added_com_range_x = [-0.05, 0.05]
    env_cfg.domain_rand.added_com_range_y = [-0.05, 0.05]
    env_cfg.domain_rand.added_com_range_z = [-0.05, 0.05]
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kp_range = [0.9, 1.1]
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_Kd_range = [0.9, 1.1]
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_motor_torque_range = [0.9, 1.1]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 3 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    img_idx = 0
    x_vel = 1.5
    y_vel = 0.0
    yaw_angle_vel = 0.0
    yaw_heading = 0.0

    for i in range(10*int(env.max_episode_length)):

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # set commands
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        if env.cfg.commands.heading_command:
            env.commands[:, 3] = yaw_heading
        else:
            env.commands[:, 2] = yaw_angle_vel

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(env.base_pos[robot_index, :].to(device="cpu"))
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': env.dof_pos_ref[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel_target': env.dof_vel_ref[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
