import numpy as np
import mujoco
from deploy.deploy_mujoco.utils.math import quat_mul

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

        # inverse quaternion (wxyz)
        q_inv = np.array([curr_quat[0], -curr_quat[1], -curr_quat[2], -curr_quat[3]])

        # orientation error
        q_err = quat_mul(target_quat, q_inv)
        rot_err = 2.0 * q_err[1:]

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