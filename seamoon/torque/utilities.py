import numpy as np
import itertools

# define epsilon
epsilon = np.zeros((3, 3, 3))
epsilon[0, 1, 2], epsilon[2, 0, 1], epsilon[1, 2, 0] = 1, 1, 1
epsilon[0, 2, 1], epsilon[1, 0, 2], epsilon[2, 1, 0] = -1, -1, -1


def construct_torque_operator(r, v):
    # r and v are N*3 numpy arrays
    N = r.shape[0]
    assert (
        v.shape[0] == N
    ), f"r is of dimensions {r.shape} and v of dimensions {v.shape}"
    T = np.zeros((3, 3, 3))
    indices = list(itertools.product(range(3), range(3), range(3)))

    for m in range(3):
        for p, j, q in indices:
            for i in range(N):
                T[m, j, p] += epsilon[p, q, m] * v[i, j] * r[i, q]
    return T


def apply_rotation(R, v):
    rotated_mode = np.zeros_like(v)
    for i in range(v.shape[0]):
        rotated_mode[i] = R @ v[i]
    return rotated_mode


def compute_angle_norm(R):
    eigenvalues, eigenvectors = np.linalg.eig(R)
    if eigenvalues[1] != 1.0:
        return np.arccos(np.real(eigenvalues[1]))
    else:
        return np.arccos(np.real(eigenvalues[0]))


def compute_torque(R, v, r):
    torque = 0
    for i in range(v.shape[0]):
        torque += np.cross(R @ v[i], r[i])
    return torque


def compute_structure_matrix_operator(H, v):
    T = np.zeros((9, 9))
    H_list = []
    N = v.shape[0]
    for i in range(N):
        H_list.append([])
        for j in range(N):
            H_list[-1].append(H[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)])

    rotation_indices = itertools.product(range(3), range(3), range(3), range(3))
    print(v.shape)
    for i in range(N):
        for j in range(N):
            x = np.expand_dims(v[i], 1) @ np.expand_dims(v[j], 0)
            T += np.kron(H_list[i][j], x)
    return T
