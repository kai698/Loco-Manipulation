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

def quat_rotate_inverse(q, v):
    q_w = q[0]
    q_vec = q[1:]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = 2.0 * q_w * np.cross(q_vec, v)
    c = 2.0 * np.dot(q_vec, v) * q_vec

    return a - b + c

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def quat_mul(q1, q2):
    """
    quaternion multiplication (wxyz format)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def sphere2cart(s):
    """Convert spherical (l, pitch, yaw) to Cartesian."""
    l, pitch, yaw = s
    x = l * np.cos(pitch) * np.cos(yaw)
    y = l * np.cos(pitch) * np.sin(yaw)
    z = l * np.sin(pitch)
    return np.array([x, y, z])

def quat_from_euler_xyz(roll, pitch, yaw):
    """
    Convert Euler XYZ to quaternion [w,x,y,z]
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])