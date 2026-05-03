import numpy as np


class EETrackingLogger:
    def __init__(self, start_time_s=1.0, end_time_s=10.0):
        if end_time_s <= start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")

        self.start_time_s = float(start_time_s)
        self.end_time_s = float(end_time_s)

        self.timestamps = []
        self.actual_positions = []
        self.target_positions = []
        self.error_vectors = []

        self._printed = False

    def log_sample(self, sim_time_s, actual_pos, target_pos):
        sim_time_s = float(sim_time_s)
        if sim_time_s < self.start_time_s or sim_time_s > self.end_time_s:
            return

        actual_pos = np.asarray(actual_pos, dtype=np.float64).reshape(3)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        error_vec = actual_pos - target_pos

        self.timestamps.append(sim_time_s)
        self.actual_positions.append(actual_pos.copy())
        self.target_positions.append(target_pos.copy())
        self.error_vectors.append(error_vec.copy())

    def maybe_print(self, sim_time_s):
        if self._printed or sim_time_s < self.end_time_s:
            return

        self.print_summary()
        self._printed = True

    def print_summary(self):
        if not self.error_vectors:
            print(
                f"No end-effector tracking samples were recorded in "
                f"[{self.start_time_s:.1f}s, {self.end_time_s:.1f}s]."
            )
            return

        errors = np.asarray(self.error_vectors, dtype=np.float64)
        targets = np.asarray(self.target_positions, dtype=np.float64)

        abs_errors = np.abs(errors)
        error_norms = np.linalg.norm(errors, axis=1)
        target_norms = np.linalg.norm(targets, axis=1)
        relative_errors = error_norms / np.maximum(target_norms, 1e-8) * 100.0

        print(
            f"EE tracking error stats from {self.start_time_s:.1f}s to "
            f"{self.end_time_s:.1f}s ({len(self.timestamps)} samples):"
        )

        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            axis_errors = abs_errors[:, axis_idx]
            print(
                f"MAE ee position {axis_name}: "
                f"{np.mean(axis_errors):.6f} ± {np.std(axis_errors):.6f} m "
                f"[min {np.min(axis_errors):.6f}, max {np.max(axis_errors):.6f}]"
            )

        print(
            f"MAE ee position norm: "
            f"{np.mean(error_norms):.6f} ± {np.std(error_norms):.6f} m "
            f"[min {np.min(error_norms):.6f}, max {np.max(error_norms):.6f}]"
        )
        print(
            f"Relative ee position error: "
            f"{np.mean(relative_errors):.2f} ± {np.std(relative_errors):.2f}% "
            f"[min {np.min(relative_errors):.2f}%, max {np.max(relative_errors):.2f}%]"
        )
