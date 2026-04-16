import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend(loc='upper right')
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend(loc='upper right')
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend(loc='upper right')
        # plot joint positions
        a = axs[1, 0]
        if log["dof_pos"]: 
            dof_pos = np.array(log["dof_pos"])
            dof_pos_limits = np.array(log["dof_pos_limits"])
            for i in range(dof_pos.shape[1]):
                a.plot(time, dof_pos[:, i], label=f'position {i}')
        a.axhline(y=dof_pos_limits[0, 0], color='r', linestyle='--', linewidth=1, label='limit')
        a.axhline(y=dof_pos_limits[0, 1], color='r', linestyle='--', linewidth=1)
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend(loc='upper right')
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: 
            dof_vel = np.array(log["dof_vel"])
            dof_vel_limits = np.array(log["dof_vel_limits"])
            for i in range(dof_vel.shape[1]):
                a.plot(time, dof_vel[:, i], label=f'velocity {i}')
        a.axhline(y=-dof_vel_limits[0], color='r', linestyle='--', linewidth=1, label='limit')
        a.axhline(y=dof_vel_limits[0], color='r', linestyle='--', linewidth=1)
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend(loc='upper right')
        # plot dof torque
        a = axs[1, 2]
        if log["torque"]:
            torque = np.array(log["torque"])
            torque_limits = np.array(log["torque_limits"])
            for i in range(torque.shape[1]):
                a.plot(time, torque[:, i], label=f'torque {i}')
        a.axhline(y=-torque_limits[0], color='r', linestyle='--', linewidth=1, label='limit')
        a.axhline(y=torque_limits[0], color='r', linestyle='--', linewidth=1)
        a.set(xlabel='time [s]', ylabel='Torque [Nm]', title='Joint Torque')
        a.legend(loc='upper right')
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            max_contact_force = np.array(log["max_contact_force"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.axhline(y=max_contact_force[0], color='r', linestyle='--', linewidth=1, label='limit')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces', ylim=(0, 200))
        a.legend(loc='upper right')
        # plot base orientation
        a = axs[2, 1]
        if log["base_orn"]: a.plot(time, log["base_orn"], label='measured')
        a.axhline(y=0.1, color='r', linestyle='--', linewidth=1, label='limit')
        a.axhline(y=-0.1, color='r', linestyle='--', linewidth=1)
        a.set(xlabel='time [s]', ylabel='base orientation [rad]', title='Base Orientation')
        a.legend(loc='upper right')
        # plot ee pos
        a = axs[2, 2]
        if log["ee_pos"]: a.plot(time, log["ee_pos"], label='measured')
        if log["ee_goal_pos"]: a.plot(time, log["ee_goal_pos"], label='commanded')
        a.set(xlabel='time [s]', ylabel='ee goal pos [m]', title='EE Goal Position')
        a.legend(loc='upper right')
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()