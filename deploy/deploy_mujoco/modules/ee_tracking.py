import numpy as np
import mujoco
from deploy.deploy_mujoco.utils.math import quat_mul, quat_from_euler_xyz, sphere2cart

class DLSIKController:
    def __init__(self, model, data, num_dof, site_name, damping=0.05, step_size=0.1):
        self.model = model
        self.data = data
        self.num_dof = num_dof
        self.site_name = site_name
        self.lmbda = damping
        self.alpha = step_size

    def get_ee_jacobian(self, site_name):
        # update kinematics
        mujoco.mj_forward(self.model, self.data)

        # get site id
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        # compute jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

        # stack → (6, nv) and slice arm part
        J = np.vstack([jacp, jacr])
        return J[:, -self.num_dof:]

    def compute_error(self, target_pos, target_quat, curr_pos, curr_quat):
        # position error
        pos_err = target_pos - curr_pos

        # normalize quaternions
        target_quat = target_quat / np.linalg.norm(target_quat)
        curr_quat = curr_quat / np.linalg.norm(curr_quat)

        # inverse quaternion (wxyz)
        q_inv = np.array([curr_quat[0], -curr_quat[1], -curr_quat[2], -curr_quat[3]])

        # orientation error
        q_err = quat_mul(target_quat, q_inv)
        rot_err = 2.0 * q_err[1:] * np.sign(q_err[0])

        return np.concatenate([pos_err, rot_err])

    def solve(self, target_pos, target_quat, curr_pos, curr_quat, q_curr):
        # update state
        mujoco.mj_forward(self.model, self.data)

        # jacobian
        J = self.get_ee_jacobian(self.site_name)

        # task-space error
        e = self.compute_error(target_pos, target_quat, curr_pos, curr_quat)

        # DLS pseudo-inverse
        JT = J.T
        JJt = J @ JT
        J_pinv = JT @ np.linalg.inv(JJt + (self.lmbda**2) * np.eye(6))

        # joint update
        dq = self.alpha * (J_pinv @ e)

        # return next joint state
        return q_curr + dq

class EEGoalSampler:
    def __init__(self, goal_ee_cfg):
        # Parameters
        self.traj_timesteps = goal_ee_cfg.traj_timesteps
        self.hold_timesteps = goal_ee_cfg.hold_timesteps
        self.traj_total_timesteps = self.traj_timesteps + self.hold_timesteps
        self.goal_ee_ranges = goal_ee_cfg.ranges

        # State
        self.goal_timer = 0.0

        self.ee_start_sphere = np.zeros(3)
        self.ee_goal_sphere = np.zeros(3)
        self.curr_ee_goal_sphere = np.zeros(3)
        self.ee_start_sphere[:] = goal_ee_cfg.ranges.init_pos_start[:]
        self.ee_goal_sphere[:] = goal_ee_cfg.ranges.init_pos_end[:]
        self.ee_start_cart = sphere2cart(self.ee_start_sphere)
        self.ee_goal_cart = sphere2cart(self.ee_goal_sphere)
        self.curr_ee_goal_cart = self.ee_start_cart.copy()

        self.default_ee_rpy = goal_ee_cfg.ranges.default_ee_rpy
        self.ee_goal_orn_quat = quat_from_euler_xyz(self.default_ee_rpy[0], self.default_ee_rpy[1], self.default_ee_rpy[2])  # wxyz
        self.ee_goal_orn_delta_rpy = np.zeros(3)

    def _update_curr_goal(self):
        """
        Interpolate goal and update position and orientation.
        """
        t = np.clip(self.goal_timer / self.traj_timesteps, 0.0, 1.0)

        # Linear interpolation in spherical space
        self.curr_ee_goal_sphere = ((1 - t) * self.ee_start_sphere + t * self.ee_goal_sphere)

        # Convert to Cartesian
        self.ee_goal_cart = sphere2cart(self.ee_goal_sphere)
        self.curr_ee_goal_cart = sphere2cart(self.curr_ee_goal_sphere)

        # ===== Orientation =====
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[0] + self.default_ee_rpy[0],
            self.ee_goal_orn_delta_rpy[1] + self.default_ee_rpy[1],
            self.ee_goal_orn_delta_rpy[2] + self.default_ee_rpy[2],
        )

        # Advance time
        self.goal_timer += 1

        if self.goal_timer > self.traj_total_timesteps:
            self._resample_goal()

    def _resample_goal(self):
        """
        Sample a new goal in spherical space.
        """
        self.ee_start_sphere = self.curr_ee_goal_sphere.copy()
        self.ee_start_cart = sphere2cart(self.ee_start_sphere)

        self._resample_sphere()
        self._resample_orientation()

        self.goal_timer = 0.0

    def _resample_sphere(self):
        """Sample position in spherical coordinates."""
        self.ee_goal_sphere[0] = np.random.uniform(self.goal_ee_ranges.pos_l[0], self.goal_ee_ranges.pos_l[1])
        self.ee_goal_sphere[1] = np.random.uniform(self.goal_ee_ranges.pos_p[0], self.goal_ee_ranges.pos_p[1])
        self.ee_goal_sphere[2] = np.random.uniform(self.goal_ee_ranges.pos_y[0], self.goal_ee_ranges.pos_y[1])

    def _resample_orientation(self):
        """Sample orientation perturbation (RPY)."""
        self.ee_goal_orn_delta_rpy[0] = np.random.uniform(self.goal_ee_ranges.delta_orn_r[0], self.goal_ee_ranges.delta_orn_r[1])
        self.ee_goal_orn_delta_rpy[1] = np.random.uniform(self.goal_ee_ranges.delta_orn_p[0], self.goal_ee_ranges.delta_orn_p[1])
        self.ee_goal_orn_delta_rpy[2] = np.random.uniform(self.goal_ee_ranges.delta_orn_y[0], self.goal_ee_ranges.delta_orn_y[1])

class EEPointVisualizer:
    VALID_POINTS = ("start", "actual", "target")
    POINT_COLORS = {
        "start": [0, 1, 0, 0.8],
        "actual": [1, 0, 0, 0.8],
        "target": [0, 0, 1, 0.6],
    }

    def __init__(self, enabled_points=("start", "actual", "target"), point_size=0.02):
        invalid_points = set(enabled_points) - set(self.VALID_POINTS)
        if invalid_points:
            raise ValueError(f"Unsupported point types: {sorted(invalid_points)}")

        self.enabled_points = tuple(enabled_points)
        self.point_size = point_size
        self.positions = {point_type: None for point_type in self.VALID_POINTS}

    def set_point(self, point_type, pos):
        if point_type not in self.VALID_POINTS:
            raise ValueError(f"Unsupported point type: {point_type}")
        self.positions[point_type] = np.asarray(pos, dtype=np.float64).copy()

    def _render_point(self, scn, pos, rgba):
        if pos is None or scn.ngeom >= scn.maxgeom:
            return

        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.point_size, self.point_size, self.point_size],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        scn.ngeom += 1

    def render(self, viewer):
        scn = viewer.user_scn
        scn.ngeom = 0

        for point_type in self.enabled_points:
            self._render_point(scn, self.positions[point_type], self.POINT_COLORS[point_type])
