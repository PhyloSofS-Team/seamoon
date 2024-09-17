from wolframclient.language import wl, wlexpr
from scipy.spatial.transform import Rotation as R
import numpy as np
import itertools
from seamoon.torque.utilities import construct_torque_operator


def format_wolfram_res(res):
    res = str(res)
    # var_name = res[res.index('`')+1:res.index(',')]
    value = float(res[res.index(",") + 2 : res.index("]")])
    return value


def quaternion_to_rotation(q):
    q0, q1, q2, q3 = q
    return np.array(
        [
            [
                2 * (q0 * q0 + q1 * q1) - 1,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                2 * (q0 * q0 + q2 * q2) - 1,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                2 * (q0 * q0 + q3 * q3) - 1,
            ],
        ]
    )


def solve_linear_in_quartenions_wolfram(T, session):
    command = "NSolve[ (-2*q1*q2 + 2*q3*q4)*T[0, 2, 1] + (2*q1*q2 + 2*q3*q4)*T[0, 1, 2] + (-2*q1*q3 + 2*q2*q4)*T[0, 0, 2] + (2*q1*q3 + 2*q2*q4)*T[0, 2, 0] + (-2*q1*q4 + 2*q2*q3)*T[0, 1, 0] + (2*q1*q4 + 2*q2*q3)*T[0, 0, 1] + (q1^2 - q2^2 - q3^2 + q4^2)*T[0, 2, 2] + (q1^2 - q2^2 + q3^2 - q4^2)*T[0, 1, 1] + (q1^2 + q2^2 - q3^2 - q4^2)*T[0, 0, 0] == 0 && (-2*q1*q2 + 2*q3*q4)*T[1, 2, 1] + (2*q1*q2 + 2*q3*q4)*T[1, 1, 2] + (-2*q1*q3 + 2*q2*q4)*T[1, 0, 2] + (2*q1*q3 + 2*q2*q4)*T[1, 2, 0] + (-2*q1*q4 + 2*q2*q3)*T[1, 1, 0] + (2*q1*q4 + 2*q2*q3)*T[1, 0, 1] + (q1^2 - q2^2 - q3^2 + q4^2)*T[1, 2, 2] + (q1^2 - q2^2 + q3^2 - q4^2)*T[1, 1, 1] + (q1^2 + q2^2 - q3^2 - q4^2)*T[1, 0, 0] == 0 && (-2*q1*q2 + 2*q3*q4)*T[2, 2, 1] + (2*q1*q2 + 2*q3*q4)*T[2, 1, 2] + (-2*q1*q3 + 2*q2*q4)*T[2, 0, 2] + (2*q1*q3 + 2*q2*q4)*T[2, 2, 0] + (-2*q1*q4 + 2*q2*q3)*T[2, 1, 0] + (2*q1*q4 + 2*q2*q3)*T[2, 0, 1] + (q1^2 - q2^2 - q3^2 + q4^2)*T[2, 2, 2] + (q1^2 - q2^2 + q3^2 - q4^2)*T[2, 1, 1] + (q1^2 + q2^2 - q3^2 - q4^2)*T[2, 0, 0] == 0 && q1^2+q2^2+q3^2+q4^2 == 1 , {q1, q2, q3, q4}]"
    indices_T = list(itertools.product(range(3), range(3), range(3)))
    for i, j, k in indices_T:
        command = command.replace(f"T[{i}, {j}, {k}]", str(T[i, j, k]))

    res = session.evaluate(wlexpr(command))
    quaternions = [
        np.array([format_wolfram_res(res[i][j]) for j in range(len(res[i]))])
        for i in range(len(res))
    ]
    solutions = [quaternion_to_rotation(q) for q in quaternions]
    return solutions, quaternions


def get_unique_rotations(r, mode, session):
    operator_mode = construct_torque_operator(r, mode)
    solutions, quaternions = solve_linear_in_quartenions_wolfram(operator_mode, session)
    unique_solutions = []
    indices_solutions = []
    for i, solution in enumerate(solutions):
        use_it = True
        for solution_bis in unique_solutions:
            if np.linalg.norm(solution - solution_bis) < 1e-3:
                use_it = False
        if use_it:
            unique_solutions.append(solution)
            indices_solutions.append(i)
    return unique_solutions
