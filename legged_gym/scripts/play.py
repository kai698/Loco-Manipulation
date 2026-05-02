from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.math import orientation_error
from rsl_rl.modules.actor_critic import Actor, get_activation

import numpy as np
import torch
import torch.nn as nn

class ExportActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.priv_encoder = actor.priv_encoder
        self.history_encoder = actor.history_encoder
        self.backbone = actor.actor_backbone
        self.leg_head = actor.actor_leg_control_head
        self.arm_head = actor.actor_arm_control_head

    def forward(self, input):
        latent = self.backbone(input)
        leg_output = self.leg_head(latent)
        arm_output = self.arm_head(latent)
        output = torch.cat([leg_output, arm_output], dim=-1)
        return output

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.8, 1.0]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.restitution_range = [0.0, 0.3]
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [-1., 1.]
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.added_com_range_x = [-0.05, 0.05]
    env_cfg.domain_rand.added_com_range_y = [-0.05, 0.05]
    env_cfg.domain_rand.added_com_range_z = [-0.05, 0.05]
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.gripper_added_mass_range = [0, 0.1]
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.leg_motor_strength_range = [0.9, 1.1]
    env_cfg.domain_rand.arm_motor_strength_range = [0.9, 1.1]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device, stochastic=args.stochastic)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if EXPORT_ACTOR_MODEL:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, 'policies', 'model_actor.pt')
        torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), save_path)
        print('Saved actor model to: ', save_path)
        save(env_cfg, train_cfg, path, save_path)

    if args.use_jit:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'traced', "traced_actor.pt")
        print("Loading jit for policy: ", path)
        policy = torch.jit.load(path, map_location=ppo_runner.device)
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'traced', "traced_hist_encoder.pt")
        history_encoder = torch.jit.load(path, map_location=ppo_runner.device)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = list(range(env.num_dof)) # all joints, including wheels
    start_state_log = 100 # number of steps before starting error statistics
    stop_state_log = 800 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    img_idx = 0
    x_vel = 1.0
    y_vel = 0.0
    yaw_angle_vel = 0.0
    yaw_heading = 0.0

    for i in range(10*int(env.max_episode_length)):

        if args.use_jit:
            latent = history_encoder(obs[:, env_cfg.env.num_proprio + env_cfg.env.num_priv:])
            actions = policy(torch.cat((obs[:, :env_cfg.env.num_proprio], latent), dim=1))
        else:
            actions = policy(obs.detach(), hist_encoding=True)

        obs, _, leg_rews, arm_rews, costs, dones, infos = env.step(actions.detach())

        # set commands
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        if env.cfg.commands.heading_command:
            env.commands[:, 3] = yaw_heading
        else:
            env.commands[:, 2] = yaw_angle_vel

        if RECORD_FRAMES:
            frames_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            if i % int(env.max_episode_length / 20) == 0 and i < stop_state_log: # save frames only for the first episode
                filename = os.path.join(frames_dir, f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(env.base_pos[robot_index, :].to(device="cpu"))
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            ee_orn_error = orientation_error(
                env.ee_goal_orn_quat[robot_index:robot_index + 1],
                env.ee_orn[robot_index:robot_index + 1]
            )
            ee_orn_error = 2.0 * torch.asin(torch.clamp(torch.norm(ee_orn_error), 0.0, 1.0)) # convert to angle error in radians
            
            logger.log_states(
                {
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].cpu().numpy(),
                    'dof_pos_limits': env.dof_pos_limits[joint_index, :].cpu().numpy(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].cpu().numpy(),
                    'dof_vel_limits': env.dof_vel_limits[joint_index].cpu().numpy(),
                    'torque': env.torques[robot_index, joint_index].cpu().numpy(),
                    'torque_limits': env.torque_limits[joint_index].cpu().numpy(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'max_contact_force': env.cfg.rewards.max_contact_force,
                    'base_orn': torch.norm(env.base_euler[robot_index, :2]).item(),
                    'ee_pos': torch.norm(env.ee_pos_local[robot_index]).item(),
                    'ee_goal_pos': torch.norm(env.ee_goal_local_cart[robot_index]).item(),
                    'ee_pos_vec': env.ee_pos_local[robot_index].cpu().numpy(),
                    'ee_goal_pos_vec': env.ee_goal_local_cart[robot_index].cpu().numpy(),
                    'ee_orn_error': ee_orn_error.item()
                }
            )
        elif i==stop_state_log:
            logger.print_tracking_errors(start_state_log)
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

def save(env_cfg, train_cfg, export_root, load_path):

    # Actor
    actor = Actor(
        env_cfg.env.num_proprio,
        train_cfg.policy.actor_hidden_dims,
        get_activation(train_cfg.policy.activation),
        train_cfg.policy.leg_control_head_hidden_dims,
        train_cfg.policy.arm_control_head_hidden_dims,
        env_cfg.env.num_leg_actions,
        env_cfg.env.num_arm_actions,
        env_cfg.env.num_priv,
        env_cfg.env.history_len, 
        env_cfg.env.num_proprio,
        train_cfg.policy.priv_encoder_dims)
    actor.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    actor.eval()

    # ExportActor
    export_actor = ExportActor(actor)
    export_actor.eval()

    # root path
    save_root = os.path.join(export_root, "traced")
    os.makedirs(save_root, exist_ok=True)

    # Save the traced actor
    dummy_actor_input = torch.zeros(1, train_cfg.policy.priv_encoder_dims[-1] + env_cfg.env.num_proprio)
    with torch.no_grad():
        export_actor(dummy_actor_input)
        traced_actor = torch.jit.trace(export_actor, dummy_actor_input)
    save_path = os.path.join(save_root, "traced_actor.pt")
    traced_actor.save(save_path)
    print(f"Saved traced actor model to: {save_path}")

    # Save the traced history encoder
    dummy_hist_input = torch.zeros(1, env_cfg.env.history_len * env_cfg.env.num_proprio)
    with torch.no_grad():
        export_actor.history_encoder(dummy_hist_input)
        traced_hist_encoder = torch.jit.trace(export_actor.history_encoder, dummy_hist_input)
    save_path = os.path.join(save_root, "traced_hist_encoder.pt")
    traced_hist_encoder.save(save_path)
    print(f"Saved traced history encoder model to: {save_path}")

if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_ACTOR_MODEL = True
    RECORD_FRAMES = True
    MOVE_CAMERA = True
    args = get_args()
    play(args)
