import numpy as np

def euler_from_quat(quat: np.ndarray):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def quat_apply(a, b):
    xyz = a[1:]
    t = np.cross(xyz, b) * 2
    return (b + a[0] * t + np.cross(xyz, t))

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles