import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process


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

    def print_tracking_errors(self, start_idx=0):
        base_vel_x = np.asarray(self.state_log["base_vel_x"][start_idx:], dtype=np.float64)
        base_vel_y = np.asarray(self.state_log["base_vel_y"][start_idx:], dtype=np.float64)
        base_vel_yaw = np.asarray(self.state_log["base_vel_yaw"][start_idx:], dtype=np.float64)
        command_x = np.asarray(self.state_log["command_x"][start_idx:], dtype=np.float64)
        command_y = np.asarray(self.state_log["command_y"][start_idx:], dtype=np.float64)
        command_yaw = np.asarray(self.state_log["command_yaw"][start_idx:], dtype=np.float64)
        ee_pos_vec = np.asarray(self.state_log["ee_pos_vec"][start_idx:], dtype=np.float64)
        ee_goal_pos_vec = np.asarray(self.state_log["ee_goal_pos_vec"][start_idx:], dtype=np.float64)
        ee_orn_error = np.asarray(self.state_log["ee_orn_error"][start_idx:], dtype=np.float64)

        x_vel_error = np.abs(base_vel_x - command_x)
        y_vel_error = np.abs(base_vel_y - command_y)
        yaw_vel_error = np.abs(base_vel_yaw - command_yaw)
        ee_pos_error = np.mean(np.abs(ee_pos_vec - ee_goal_pos_vec), axis=1)
        ee_orn_abs_error = np.abs(ee_orn_error)

        print(f"MAE x linear velocity: {np.mean(x_vel_error):.6f} ± {np.std(x_vel_error):.6f} [min {np.min(x_vel_error):.6f}, max {np.max(x_vel_error):.6f}]")
        print(f"MAE y linear velocity: {np.mean(y_vel_error):.6f} ± {np.std(y_vel_error):.6f} [min {np.min(y_vel_error):.6f}, max {np.max(y_vel_error):.6f}]")
        print(f"MAE z angular velocity: {np.mean(yaw_vel_error):.6f} ± {np.std(yaw_vel_error):.6f} [min {np.min(yaw_vel_error):.6f}, max {np.max(yaw_vel_error):.6f}]")
        print(f"MAE ee position: {np.mean(ee_pos_error):.6f} ± {np.std(ee_pos_error):.6f} [min {np.min(ee_pos_error):.6f}, max {np.max(ee_pos_error):.6f}]")
        print(f"MAE ee orientation: {np.mean(ee_orn_abs_error):.6f} ± {np.std(ee_orn_abs_error):.6f} [min {np.min(ee_orn_abs_error):.6f}, max {np.max(ee_orn_abs_error):.6f}]")

    def _time_axis(self):
        for value in self.state_log.values():
            return np.linspace(0, len(value) * self.dt, len(value))
        return np.array([])

    def _plot_series(self, ax, time, actual_key, expected_key=None, title='', ylabel='', actual_label='measured', expected_label='commanded', limit=None, lower_limit=None, ylim=None):
        log = self.state_log
        if actual_key in log and log[actual_key]:
            actual = np.asarray(log[actual_key])
            if actual.ndim == 1:
                ax.plot(time, actual, label=actual_label)
            else:
                for i in range(actual.shape[1]):
                    ax.plot(time, actual[:, i], label=f'{actual_label} {i}')
        if expected_key is not None and expected_key in log and log[expected_key]:
            expected = np.asarray(log[expected_key])
            if expected.ndim == 1:
                ax.plot(time, expected, label=expected_label)
            else:
                for i in range(expected.shape[1]):
                    ax.plot(time, expected[:, i], linestyle='--', label=f'{expected_label} {i}')
        if limit is not None:
            if np.ndim(limit) == 0:
                ax.axhline(y=limit, color='r', linestyle='--', linewidth=1)
            else:
                limit = np.asarray(limit)
                if limit.ndim == 1:
                    ax.axhline(y=limit[0], color='r', linestyle='--', linewidth=1)
                    ax.axhline(y=limit[1], color='r', linestyle='--', linewidth=1)
                elif limit.ndim == 2:
                    for i in range(limit.shape[0]):
                        ax.axhline(y=limit[i, 0], color='r', linestyle='--', linewidth=1)
                        ax.axhline(y=limit[i, 1], color='r', linestyle='--', linewidth=1)
        if lower_limit is not None:
            ax.axhline(y=lower_limit, color='r', linestyle='--', linewidth=1)
        if ylim is not None:
            ax.set_ylim(**ylim)
        ax.set(xlabel='time [s]', ylabel=ylabel, title=title)
        ax.legend(loc='lower right')

    def _get_joint_limits(self, data_key, limit_key):
        log = self.state_log
        if limit_key not in log or not log[limit_key]:
            return None
        limits = np.asarray(log[limit_key])
        if limits.ndim == 3:
            limits = limits[0]
        elif limits.ndim == 2 and data_key != 'dof_pos':
            limits = limits[0]

        if data_key == 'dof_pos':
            return limits

        upper = np.asarray(limits)
        return np.stack((-upper, upper), axis=1)

    def _plot_joint_groups(self, axs_row, data_key, limit_key, ylabel, actual_label):
        log = self.state_log
        time = self._time_axis()
        if data_key not in log or not log[data_key]:
            return
        data = np.asarray(log[data_key])
        limits = self._get_joint_limits(data_key, limit_key)
        joint_groups = [
            ("Hip Joints", [0, 4, 8, 12]),
            ("Thigh Joints", [1, 5, 9, 13]),
            ("Calf Joints", [2, 6, 10, 14]),
            ("Wheel Joints", [3, 7, 11, 15]),
        ]
        leg_labels = ["FL", "FR", "RL", "RR"]
        for ax, (group_title, indices) in zip(axs_row, joint_groups):
            for leg_label, idx in zip(leg_labels, indices):
                ax.plot(time, data[:, idx], label=f'{actual_label} {leg_label}')
                if limits is not None:
                    ax.axhline(
                        y=limits[idx, 0],
                        color='r',
                        linestyle='--',
                        linewidth=1,
                    )
                    ax.axhline(
                        y=limits[idx, 1],
                        color='r',
                        linestyle='--',
                        linewidth=1,
                    )
            ax.set(xlabel='time [s]', ylabel=ylabel, title=group_title)
            ax.legend(loc='lower right')

    def _plot(self):
        time = self._time_axis()
        log = self.state_log

        fig1, axs1 = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
        self._plot_series(axs1[0], time, 'base_vel_x', 'command_x', 'Base Velocity X', 'base lin vel [m/s]')
        self._plot_series(axs1[1], time, 'base_vel_y', 'command_y', 'Base Velocity Y', 'base lin vel [m/s]')
        self._plot_series(axs1[2], time, 'base_vel_yaw', 'command_yaw', 'Base Velocity Yaw', 'base ang vel [rad/s]')

        fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        self._plot_series(axs2[0], time, 'ee_pos_vec', 'ee_goal_pos_vec', 'EE Position', 'ee pos [m]')
        self._plot_series(axs2[1], time, 'ee_pos', 'ee_goal_pos', 'EE Position Norm', 'ee pos norm [m]')

        fig3, axs3 = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        self._plot_series(axs3[0], time, 'contact_forces_z', None, 'Foot Contact Forces', 'force z [N]', limit=log.get('max_contact_force', None), lower_limit=0.0, ylim={'top': 250})
        self._plot_series(axs3[1], time, 'base_orn', None, 'Base Orientation Norm', 'base orientation [rad]', limit=0.1, lower_limit=0.0)

        fig4, axs4 = plt.subplots(3, 4, figsize=(24, 16), constrained_layout=True)
        self._plot_joint_groups(axs4[0], 'dof_pos', 'dof_pos_limits', 'position [rad]', 'measured')
        self._plot_joint_groups(axs4[1], 'dof_vel', 'dof_vel_limits', 'velocity [rad/s]', 'measured')
        self._plot_joint_groups(axs4[2], 'torque', 'torque_limits', 'torque [Nm]', 'measured')
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
