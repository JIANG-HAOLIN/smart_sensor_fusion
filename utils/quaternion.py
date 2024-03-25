import numpy as np
from scipy.interpolate import splrep, BSpline

"""
DEFINE QUATERNION COMPUTATIONS AND MAPPINGS
the logarithmic and exponential maps for quaternion
    q_log_map(p, base=None)
    q_exp_map(p, base=None)
    q_parallel_transport(p, g, h)
and some basic calculations
    q_mul(q1, q2)
    q_inverse(q)
    q_div(q1, q2)
    q_norm_squared(q)
    q_norm(q)
    q_to_rotation_matrix(q)
    q_to_quaternion_matrix(q)
    q_to_euler(q)
    q_from_rot_mat(rot_mat)
    q_from_euler(euler)
    q_to_intrinsic_xyz(q)
    q_from_intrinsic_xyz(euler)
"""


def q_exp_map(v, base=None):
    """
    The exponential quaternion map maps v from the tangent space at base to q on the manifold
    S^3. See Table 2.1 in reference:
    [7] "Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018.

    Parameters
    ----------
    :param v: np,array of shape (3, N)

    Optional parameters
    -------------------
    :param base: np,array of shape (4,), the base quaternion. If None the neutral element [1, 0, 0, 0] is used.

    Returns
    -------
    :return q: np.array of shape (4, N), the N quaternions corresponding to the N vectors in v
    """
    v_2d = v.reshape((3, 1)) if len(v.shape) == 1 else v
    if base is None:
        norm_v = np.sqrt(np.sum(v_2d ** 2, 0))
        q = np.append(np.ones((1, v_2d.shape[1])), np.zeros((3, v_2d.shape[1])), 0)
        non_0 = np.where(norm_v > 0)[0]
        q[:, non_0] = np.append(
            np.cos(norm_v[non_0]).reshape((1, non_0.shape[0])),
            np.tile(np.sin(norm_v[non_0]) / norm_v[non_0], (3, 1)) * v_2d[:, non_0],
            0,
        )
        return q.reshape(4) if len(v.shape) == 1 else q
    else:
        return q_mul(base, q_exp_map(v))


def q_log_map(q, base=None):
    """
    The logarithmic quaternion map maps q from the manifold S^3 to v in the tangent space at base. See Table 2.1 in [7]

    Parameters
    ----------
    :param q: np,array of shape (4, N), N quaternions

    Optional parameters
    -------------------
    :param base: np,array of shape (4,), the base quaternion. If None the neutral element [1, 0, 0, 0] is used.

    Returns
    -------
    :return v: np.array of shape (3, N), the N vectors in tangent space corresponding to quaternions q
    """
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    if base is None:
        norm_q = np.sqrt(np.sum(q_2d[1:, :] ** 2, 0))
        # to avoid numeric errors for norm_q
        non_0 = np.where((norm_q > 0) * (np.abs(q_2d[0, :]) <= 1))[0]
        q_non_singular = q_2d[:, non_0]
        acos = np.arccos(q_non_singular[0, :])
        # this is *critical* for ensuring q and -q maps are the same rotation
        acos[np.where(q_non_singular[0, :] < 0)] += -np.pi
        v = np.zeros((3, q_2d.shape[1]))
        v[:, non_0] = q_non_singular[1:, :] * np.tile(acos / norm_q[non_0], (3, 1))
        if len(q.shape) == 1:
            return v.reshape(3)
        return v
    else:
        return q_log_map(q_mul(q_inverse(base), q))




def q_parallel_transport(p_g, g, h):
    """
    Transport p in tangent space at g to tangent space at h. According to (2.11)--(2.13) in [7].

    Parameters
    ----------
    :param p_g: np.array of shape (3,) point in tangent space
    :param g: np.array of shape (4,) quaternion
    :param h: np.array of shape (4,) quaternion

    Returns
    -------
    :return p_h: np.array of shape (3,) the point p_g in tangent space at h
    """
    R_e_g = q_to_quaternion_matrix(g)
    R_h_e = q_to_quaternion_matrix(h).T
    B = np.append(np.zeros((3, 1)), np.eye(3), 1).T
    log_g_h = q_log_map(h, base=g)
    m = np.linalg.norm(log_g_h)
    if m < 1e-10:
        return p_g
    u = R_e_g.dot(np.append(0, log_g_h / m)).reshape((4, 1))
    R_g_h = np.eye(4) - np.sin(m) * g.reshape((4, 1)).dot(u.T) + (np.cos(m) - 1) * u.dot(u.T)
    A_g_h = B.T.dot(R_h_e).dot(R_g_h).dot(R_e_g).dot(B)
    return A_g_h.dot(p_g)


def q_mul(q1, q2):
    return q_to_quaternion_matrix(q1).dot(q2)


def q_inverse(q):
    w0, x0, y0, z0 = q
    return np.array([w0, -x0, -y0, -z0]) / q_norm_squared(q)


def q_div(q1, q2):
    return q_mul(q1, q_inverse(q2))


def q_norm_squared(q):
    return np.sum(q ** 2)


def q_norm(q):
    return np.sqrt(q_norm_squared(q))


def q_to_rotation_matrix(q):
    """
    Computes rotation matrix out of the quaternion.

    Parameters
    ----------
    :param q: np.array of shape (4,), the quaternion

    Returns
    -------
    :return rot_mat: np.array of shape (3, 3)
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
        ]
    )


def q_to_quaternion_matrix(q):
    """
    Computes quaternion matrix out of the quaternion.

    Parameters
    ----------
    :param q: np.array of shape (4,), the quaternion

    Returns
    -------
    :return quat_mat: np.array of shape (4, 4)
    """
    return np.array(
        [[q[0], -q[1], -q[2], -q[3]], [q[1], q[0], -q[3], q[2]], [q[2], q[3], q[0], -q[1]], [q[3], -q[2], q[1], q[0]]]
    )


def q_to_euler(q):
    """
    Computes euler angles out of the quaternion. Format XYZ (roll, pitch, yaw)

    Parameters
    ----------
    :param q: np.array of shape (4, N) or (4,), the quaternion

    Returns
    -------
    :return euler: np.array of shape (3, N) or (3,)
    """
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    w, x, y, z = q_2d[0, :], q_2d[1, :], q_2d[2, :], q_2d[3, :]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    euler = np.stack([roll, pitch, yaw], axis=0)
    euler = (
        euler.reshape(
            3,
        )
        if euler.shape[1] == 1
        else euler
    )
    return euler


def q_from_rot_mat(rot_mat):
    """
    Computes the quaternion out of a given rotation matrix

    Parameters
    ----------
    :param rot_mat:  np.array of shape (3, 3)

    Returns
    -------
    :return q: np.array of shape (4,), quaternion corresponding to the rotation matrix rot_mat
    """
    qs = min(np.sqrt(np.trace(rot_mat) + 1) / 2.0, 1.0)
    kx = rot_mat[2, 1] - rot_mat[1, 2]  # Oz - Ay
    ky = rot_mat[0, 2] - rot_mat[2, 0]  # Ax - Nz
    kz = rot_mat[1, 0] - rot_mat[0, 1]  # Ny - Ox
    if (rot_mat[0, 0] >= rot_mat[1, 1]) and (rot_mat[0, 0] >= rot_mat[2, 2]):
        kx1 = rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2] + 1  # Nx - Oy - Az + 1
        ky1 = rot_mat[1, 0] + rot_mat[0, 1]  # Ny + Ox
        kz1 = rot_mat[2, 0] + rot_mat[0, 2]  # Nz + Ax
        add = kx >= 0
    elif rot_mat[1, 1] >= rot_mat[2, 2]:
        kx1 = rot_mat[1, 0] + rot_mat[0, 1]  # Ny + Ox
        ky1 = rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2] + 1  # Oy - Nx - Az + 1
        kz1 = rot_mat[2, 1] + rot_mat[1, 2]  # Oz + Ay
        add = ky >= 0
    else:
        kx1 = rot_mat[2, 0] + rot_mat[0, 2]  # Nz + Ax
        ky1 = rot_mat[2, 1] + rot_mat[1, 2]  # Oz + Ay
        kz1 = rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1] + 1  # Az - Nx - Oy + 1
        add = kz >= 0
    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1
    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = np.array([1, 0, 0, 0])
    else:
        s = np.sqrt(1 - qs ** 2) / nm
        qv = s * np.array([kx, ky, kz])
        q = np.append(qs, qv)
    return q


def q_from_euler(euler):
    """
    Computes quaternion out of the euler angles.

    Parameters
    ----------
    :param euler: np.array of shape (3, N) or (3,). Euler angles XYZ (roll, pitch, yaw)

    Returns
    -------
    :return q: np.array of shape (4, N) or (4,)
    """
    euler_2d = euler.reshape((3, 1)) if len(euler.shape) == 1 else euler
    roll, pitch, yaw = euler_2d[0, :], euler_2d[1, :], euler_2d[2, :]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.stack([w, x, y, z], axis=0)
    q = (
        q.reshape(
            4,
        )
        if q.shape[1] == 1
        else q
    )
    return q


def q_from_intrinsic_xyz(euler):
    """
    Convert euler angle as intrinsic rotations x-y-z to quaternion.
    Note: Euler angles in "q_from_euler()" is defined as extrinsic rotations x-y-z (roll, pitch, yaw),
    or equivalently to intrinsic rotations z-y-x.

    Parameters
    ----------
    :param euler: np.array of shape (3, N) or (3,). Euler angle as intrinsic rotations x-y-z.

    Returns
    -------
    :return q: np.array of shape (4, N) or (4,)
    """
    euler_2d = euler.reshape((3, 1)) if len(euler.shape) == 1 else euler
    angle_x, angle_y, angle_z = euler_2d[0, :], euler_2d[1, :], euler_2d[2, :]
    cx = np.cos(angle_x * 0.5)
    sx = np.sin(angle_x * 0.5)
    cy = np.cos(angle_y * 0.5)
    sy = np.sin(angle_y * 0.5)
    cz = np.cos(angle_z * 0.5)
    sz = np.sin(angle_z * 0.5)

    w = cx * cy * cz - sx * sy * sz
    x = cx * sy * sz + sx * cy * cz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    q = np.stack([w, x, y, z], axis=0)
    q = (
        q.reshape(
            4,
        )
        if q.shape[1] == 1
        else q
    )
    return q


def q_to_intrinsic_xyz(q):
    """
    Computes euler angles out of the intrinsic xyz rotations.

    Parameters
    ----------
    :param q: np.array of shape (4, N) or (4,), the quaternion.

    Returns
    -------
    :return euler: np.array of shape (3, N) or (3,)
    """
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    w, x, y, z = q_2d[0, :], q_2d[1, :], q_2d[2, :], q_2d[3, :]
    t0 = 2.0 * (w * x - y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    angle_x = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y + z * x), -1.0, 1.0)
    angle_y = np.arcsin(t2)
    t3 = 2.0 * (w * z - x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    angle_z = np.arctan2(t3, t4)
    euler = np.stack([angle_x, angle_y, angle_z], axis=0)
    euler = (
        euler.reshape(
            3,
        )
        if euler.shape[1] == 1
        else euler
    )
    return euler


def compute_delta(pos, base):
    delta_p = pos[:3] - base[:3]
    delta_o = q_log_map(pos[3:], base[3:])
    return np.concatenate([delta_p, delta_o])


def compute_integral(delta, base):
    pos_p = delta[:3] + base[:3]
    pos_o = q_exp_map(delta[3:], base[3:])
    return np.concatenate([pos_p, pos_o])


def compute_sequence_integral(delta, base):
    pos_p = delta[:, :3] + base[None, :3]
    pos_o = q_exp_map(delta[:, 3:].transpose(1, 0), base[3:]).transpose(1, 0)
    return np.concatenate([pos_p, pos_o], axis=1)


def compute_sequence_delta(pos, base):
    delta_p = pos[:, :3] - base[None, :3]
    delta_o = q_log_map(pos[:, 3:].transpose(1, 0), base[3:]).transpose(1, 0)
    return np.concatenate([delta_p, delta_o], axis=1)


def recover_pose_from_relative_vel(future_vel_seq: np.ndarray, base: np.ndarray, vel_scale=0.01):
    """
    base: [7,]
    future_pose_seq: [10, 7]

    """
    recover_pose = np.zeros([future_vel_seq.shape[0], 7])
    for i in range(future_vel_seq.shape[0]):
        out = compute_integral(future_vel_seq[i, :] * vel_scale, base)
        base = out
        recover_pose[i] = out
    return recover_pose


def smooth_traj(pm: np.ndarray, s: tuple) -> (np.ndarray, np.ndarray):
    """
    pm: input trajectory, with shape [N, 7]
    """
    t = np.arange(pm.shape[0])
    pm_delta = compute_sequence_delta(pm.copy(), np.array([0, 0, 0, 0, 1, 0, 0]))
    pmr_delta = []
    for i in range(6):
        x = BSpline(*splrep(t, pm_delta[:, i], s=s[i]))(t)
        pmr_delta.append(x)
    pmr_delta = np.stack(pmr_delta, axis=1)
    pmr = compute_sequence_integral(pmr_delta, np.array([0, 0, 0, 0, 1, 0, 0]))
    return pmr, pm_delta, pmr_delta


# q1 = np.array([1,2,3,4])/np.linalg.norm([1,2,3,4],2)
# q2 = np.array([2,3,4,5])/np.linalg.norm([2,3,4,5],2)
# q3 = np.array([3,4,5,6])/np.linalg.norm([3,4,5,6],2)
# v12 = q_log_map(q2, q1)
# v23 = q_log_map(q3, q2)
#
# v1 = q_log_map(q1)
# v2 = q_log_map(q2)
# v3 = q_log_map(q3)
# v12_ = v2 - v1
# v23_ = v3 - v2
# print(v12 - v12_)
# print(v23 - v23_)
