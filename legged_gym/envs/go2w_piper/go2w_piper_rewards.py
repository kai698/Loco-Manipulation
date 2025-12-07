import torch
from isaacgym.torch_utils import *    

class Go2wPiperRewards:
    def __init__(self, env):
        self.env = env

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1) - self.env.measured_heights, dim=1)
        return torch.square(base_height - self.env.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        dof_vel = self.env.dof_vel.clone()
        dof_vel[:, self.env.wheel_indices] = 0
        return torch.sum(torch.square(dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.env.reset_buf * ~self.env.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.env.dof_vel) - self.env.dof_vel_limits * self.env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.env.torques) - self.env.torque_limits * self.env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts) 
        self.env.last_contacts = contact
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt
        rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.env.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.env.contact_forces[:, self.env.feet_indices, :2], dim=2) >\
             5 * torch.abs(self.env.contact_forces[:, self.env.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands        
        dof_err = self.env.dof_pos - self.env.default_dof_pos
        dof_err[:, self.env.wheel_indices] = 0
        return torch.norm(dof_err, dim=1) * (torch.norm(self.env.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_run_still(self):
        # Penalize motion at running commands        
        dof_err = self.env.dof_pos - self.env.default_dof_pos
        dof_err[:, self.env.wheel_indices] = 0
        return torch.norm(dof_err, dim=1) * (torch.norm(self.env.commands[:, :2], dim=1) > 0.1)
    
    def _reward_joint_power(self):
        # Penalize joint power consumption
        power = self.env.torques * self.env.dof_vel
        power[:, self.env.wheel_indices] = 0
        return torch.sum(torch.abs(power), dim=1)
    
    def _reward_joint_mirror(self):
        # Penalize difference between mirror joints
        mirror_err = torch.zeros(self.env.num_envs, device=self.env.device, requires_grad=False)
        for left_idx, right_idx in self.env.mirror_joint_indices:
            diff = torch.square(self.env.dof_pos[:, left_idx] - self.env.dof_pos[:, right_idx])
            mirror_err += diff
        mirror_err *= 1 / len(self.env.mirror_joint_indices) if len(self.env.mirror_joint_indices) > 0 else 0.
        return mirror_err