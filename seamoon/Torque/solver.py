from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
kernel = "/mnt/c/Program Files/Wolfram Research/Wolfram Engine/14.0/WolframKernel.exe"
from scipy.spatial.transform import Rotation as R
import numpy as np
import itertools
import cvxpy as cp
from qcqp import *
from utilities import construct_torque_operator
def format_wolfram_res(res):
    res = str(res)
    #var_name = res[res.index('`')+1:res.index(',')]
    value = float(res[res.index(',')+2: res.index(']')])
    return value

def quaternion_to_rotation(q):
    q0, q1, q2, q3 = q
    return np.array([[2 * (q0 * q0 + q1 * q1) - 1, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                            [2 * (q1 * q2 + q0 * q3), 2 * (q0 * q0 + q2 * q2) - 1, 2 * (q2 * q3 - q0 * q1)],
                            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0 * q0 + q3 * q3) - 1]])


def solve_linear_in_quartenions_wolfram(T, session):
    command = "NSolve[ (-2*q1*q2 + 2*q3*q4)*T[0, 2, 1] + (2*q1*q2 + 2*q3*q4)*T[0, 1, 2] + (-2*q1*q3 + 2*q2*q4)*T[0, 0, 2] + (2*q1*q3 + 2*q2*q4)*T[0, 2, 0] + (-2*q1*q4 + 2*q2*q3)*T[0, 1, 0] + (2*q1*q4 + 2*q2*q3)*T[0, 0, 1] + (q1^2 - q2^2 - q3^2 + q4^2)*T[0, 2, 2] + (q1^2 - q2^2 + q3^2 - q4^2)*T[0, 1, 1] + (q1^2 + q2^2 - q3^2 - q4^2)*T[0, 0, 0] == 0 && (-2*q1*q2 + 2*q3*q4)*T[1, 2, 1] + (2*q1*q2 + 2*q3*q4)*T[1, 1, 2] + (-2*q1*q3 + 2*q2*q4)*T[1, 0, 2] + (2*q1*q3 + 2*q2*q4)*T[1, 2, 0] + (-2*q1*q4 + 2*q2*q3)*T[1, 1, 0] + (2*q1*q4 + 2*q2*q3)*T[1, 0, 1] + (q1^2 - q2^2 - q3^2 + q4^2)*T[1, 2, 2] + (q1^2 - q2^2 + q3^2 - q4^2)*T[1, 1, 1] + (q1^2 + q2^2 - q3^2 - q4^2)*T[1, 0, 0] == 0 && (-2*q1*q2 + 2*q3*q4)*T[2, 2, 1] + (2*q1*q2 + 2*q3*q4)*T[2, 1, 2] + (-2*q1*q3 + 2*q2*q4)*T[2, 0, 2] + (2*q1*q3 + 2*q2*q4)*T[2, 2, 0] + (-2*q1*q4 + 2*q2*q3)*T[2, 1, 0] + (2*q1*q4 + 2*q2*q3)*T[2, 0, 1] + (q1^2 - q2^2 - q3^2 + q4^2)*T[2, 2, 2] + (q1^2 - q2^2 + q3^2 - q4^2)*T[2, 1, 1] + (q1^2 + q2^2 - q3^2 - q4^2)*T[2, 0, 0] == 0 && q1^2+q2^2+q3^2+q4^2 == 1 , {q1, q2, q3, q4}]"
    indices_T = list(itertools.product(range(3), range(3), range(3)))
    for i, j, k in indices_T:
        command = command.replace(f'T[{i}, {j}, {k}]', str(T[i, j, k]))
    
    res = session.evaluate(wlexpr(command))
    quaternions = [np.array([format_wolfram_res(res[i][j]) for j in range(len(res[i]))]) for i in range(len(res))]
    solutions = [quaternion_to_rotation(q) for q in quaternions]
    return solutions, quaternions

def minimize_quadratic_equation_on_rotation_wolfram(T):
    quartic_expression = "(-2*q0*q1 + 2*q2*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 5] + (2*q0*q1 + 2*q2*q3)*T[7, 5] + (-2*q0*q2 + 2*q1*q3)*T[6, 5] + (2*q0*q2 + 2*q1*q3)*T[2, 5] + (-2*q0*q3 + 2*q1*q2)*T[1, 5] + (2*q0*q3 + 2*q1*q2)*T[3, 5] + (2*q0^2 + 2*q1^2 - 1)*T[0, 5] + (2*q0^2 + 2*q2^2 - 1)*T[4, 5] + (2*q0^2 + 2*q3^2 - 1)*T[8, 5]) + (2*q0*q1 + 2*q2*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 7] + (2*q0*q1 + 2*q2*q3)*T[7, 7] + (-2*q0*q2 + 2*q1*q3)*T[6, 7] + (2*q0*q2 + 2*q1*q3)*T[2, 7] + (-2*q0*q3 + 2*q1*q2)*T[1, 7] + (2*q0*q3 + 2*q1*q2)*T[3, 7] + (2*q0^2 + 2*q1^2 - 1)*T[0, 7] + (2*q0^2 + 2*q2^2 - 1)*T[4, 7] + (2*q0^2 + 2*q3^2 - 1)*T[8, 7]) + (-2*q0*q2 + 2*q1*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 6] + (2*q0*q1 + 2*q2*q3)*T[7, 6] + (-2*q0*q2 + 2*q1*q3)*T[6, 6] + (2*q0*q2 + 2*q1*q3)*T[2, 6] + (-2*q0*q3 + 2*q1*q2)*T[1, 6] + (2*q0*q3 + 2*q1*q2)*T[3, 6] + (2*q0^2 + 2*q1^2 - 1)*T[0, 6] + (2*q0^2 + 2*q2^2 - 1)*T[4, 6] + (2*q0^2 + 2*q3^2 - 1)*T[8, 6]) + (2*q0*q2 + 2*q1*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 2] + (2*q0*q1 + 2*q2*q3)*T[7, 2] + (-2*q0*q2 + 2*q1*q3)*T[6, 2] + (2*q0*q2 + 2*q1*q3)*T[2, 2] + (-2*q0*q3 + 2*q1*q2)*T[1, 2] + (2*q0*q3 + 2*q1*q2)*T[3, 2] + (2*q0^2 + 2*q1^2 - 1)*T[0, 2] + (2*q0^2 + 2*q2^2 - 1)*T[4, 2] + (2*q0^2 + 2*q3^2 - 1)*T[8, 2]) + (-2*q0*q3 + 2*q1*q2)*((-2*q0*q1 + 2*q2*q3)*T[5, 1] + (2*q0*q1 + 2*q2*q3)*T[7, 1] + (-2*q0*q2 + 2*q1*q3)*T[6, 1] + (2*q0*q2 + 2*q1*q3)*T[2, 1] + (-2*q0*q3 + 2*q1*q2)*T[1, 1] + (2*q0*q3 + 2*q1*q2)*T[3, 1] + (2*q0^2 + 2*q1^2 - 1)*T[0, 1] + (2*q0^2 + 2*q2^2 - 1)*T[4, 1] + (2*q0^2 + 2*q3^2 - 1)*T[8, 1]) + (2*q0*q3 + 2*q1*q2)*((-2*q0*q1 + 2*q2*q3)*T[5, 3] + (2*q0*q1 + 2*q2*q3)*T[7, 3] + (-2*q0*q2 + 2*q1*q3)*T[6, 3] + (2*q0*q2 + 2*q1*q3)*T[2, 3] + (-2*q0*q3 + 2*q1*q2)*T[1, 3] + (2*q0*q3 + 2*q1*q2)*T[3, 3] + (2*q0^2 + 2*q1^2 - 1)*T[0, 3] + (2*q0^2 + 2*q2^2 - 1)*T[4, 3] + (2*q0^2 + 2*q3^2 - 1)*T[8, 3]) + (2*q0^2 + 2*q1^2 - 1)*((-2*q0*q1 + 2*q2*q3)*T[5, 0] + (2*q0*q1 + 2*q2*q3)*T[7, 0] + (-2*q0*q2 + 2*q1*q3)*T[6, 0] + (2*q0*q2 + 2*q1*q3)*T[2, 0] + (-2*q0*q3 + 2*q1*q2)*T[1, 0] + (2*q0*q3 + 2*q1*q2)*T[3, 0] + (2*q0^2 + 2*q1^2 - 1)*T[0, 0] + (2*q0^2 + 2*q2^2 - 1)*T[4, 0] + (2*q0^2 + 2*q3^2 - 1)*T[8, 0]) + (2*q0^2 + 2*q2^2 - 1)*((-2*q0*q1 + 2*q2*q3)*T[5, 4] + (2*q0*q1 + 2*q2*q3)*T[7, 4] + (-2*q0*q2 + 2*q1*q3)*T[6, 4] + (2*q0*q2 + 2*q1*q3)*T[2, 4] + (-2*q0*q3 + 2*q1*q2)*T[1, 4] + (2*q0*q3 + 2*q1*q2)*T[3, 4] + (2*q0^2 + 2*q1^2 - 1)*T[0, 4] + (2*q0^2 + 2*q2^2 - 1)*T[4, 4] + (2*q0^2 + 2*q3^2 - 1)*T[8, 4]) + (2*q0^2 + 2*q3^2 - 1)*((-2*q0*q1 + 2*q2*q3)*T[5, 8] + (2*q0*q1 + 2*q2*q3)*T[7, 8] + (-2*q0*q2 + 2*q1*q3)*T[6, 8] + (2*q0*q2 + 2*q1*q3)*T[2, 8] + (-2*q0*q3 + 2*q1*q2)*T[1, 8] + (2*q0*q3 + 2*q1*q2)*T[3, 8] + (2*q0^2 + 2*q1^2 - 1)*T[0, 8] + (2*q0^2 + 2*q2^2 - 1)*T[4, 8] + (2*q0^2 + 2*q3^2 - 1)*T[8, 8])"
    indices_T = list(itertools.product(range(9), range(9)))
    command = "Minimize[{"+quartic_expression+" , q1^2+q2^2+q3^2+q4^2==1} , {q1, q2, q3, q4}]"
    
    for i, j in indices_T:
        command = command.replace(f'T[{i}, {j}]', str(T[i, j]))
    with WolframLanguageSession(kernel) as session:
        session.start()
        res = session.evaluate(wlexpr(command))
    print(res)
    quaternions = [np.array([format_wolfram_res(res[i][j]) for j in range(len(res[i]))]) for i in range(len(res))]
    solutions = [quaternion_to_rotation(q) for q in quaternions]
    return solutions, quaternions

def minimize_quadratic_equation_on_quaternion_cpcq(operator_T):
    quartic = "(-2*q0*q1 + 2*q2*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 5] + (2*q0*q1 + 2*q2*q3)*T[7, 5] + (-2*q0*q2 + 2*q1*q3)*T[6, 5] + (2*q0*q2 + 2*q1*q3)*T[2, 5] + (-2*q0*q3 + 2*q1*q2)*T[1, 5] + (2*q0*q3 + 2*q1*q2)*T[3, 5] + (2*q0^2 + 2*q1^2 - 1)*T[0, 5] + (2*q0^2 + 2*q2^2 - 1)*T[4, 5] + (2*q0^2 + 2*q3^2 - 1)*T[8, 5]) + (2*q0*q1 + 2*q2*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 7] + (2*q0*q1 + 2*q2*q3)*T[7, 7] + (-2*q0*q2 + 2*q1*q3)*T[6, 7] + (2*q0*q2 + 2*q1*q3)*T[2, 7] + (-2*q0*q3 + 2*q1*q2)*T[1, 7] + (2*q0*q3 + 2*q1*q2)*T[3, 7] + (2*q0^2 + 2*q1^2 - 1)*T[0, 7] + (2*q0^2 + 2*q2^2 - 1)*T[4, 7] + (2*q0^2 + 2*q3^2 - 1)*T[8, 7]) + (-2*q0*q2 + 2*q1*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 6] + (2*q0*q1 + 2*q2*q3)*T[7, 6] + (-2*q0*q2 + 2*q1*q3)*T[6, 6] + (2*q0*q2 + 2*q1*q3)*T[2, 6] + (-2*q0*q3 + 2*q1*q2)*T[1, 6] + (2*q0*q3 + 2*q1*q2)*T[3, 6] + (2*q0^2 + 2*q1^2 - 1)*T[0, 6] + (2*q0^2 + 2*q2^2 - 1)*T[4, 6] + (2*q0^2 + 2*q3^2 - 1)*T[8, 6]) + (2*q0*q2 + 2*q1*q3)*((-2*q0*q1 + 2*q2*q3)*T[5, 2] + (2*q0*q1 + 2*q2*q3)*T[7, 2] + (-2*q0*q2 + 2*q1*q3)*T[6, 2] + (2*q0*q2 + 2*q1*q3)*T[2, 2] + (-2*q0*q3 + 2*q1*q2)*T[1, 2] + (2*q0*q3 + 2*q1*q2)*T[3, 2] + (2*q0^2 + 2*q1^2 - 1)*T[0, 2] + (2*q0^2 + 2*q2^2 - 1)*T[4, 2] + (2*q0^2 + 2*q3^2 - 1)*T[8, 2]) + (-2*q0*q3 + 2*q1*q2)*((-2*q0*q1 + 2*q2*q3)*T[5, 1] + (2*q0*q1 + 2*q2*q3)*T[7, 1] + (-2*q0*q2 + 2*q1*q3)*T[6, 1] + (2*q0*q2 + 2*q1*q3)*T[2, 1] + (-2*q0*q3 + 2*q1*q2)*T[1, 1] + (2*q0*q3 + 2*q1*q2)*T[3, 1] + (2*q0^2 + 2*q1^2 - 1)*T[0, 1] + (2*q0^2 + 2*q2^2 - 1)*T[4, 1] + (2*q0^2 + 2*q3^2 - 1)*T[8, 1]) + (2*q0*q3 + 2*q1*q2)*((-2*q0*q1 + 2*q2*q3)*T[5, 3] + (2*q0*q1 + 2*q2*q3)*T[7, 3] + (-2*q0*q2 + 2*q1*q3)*T[6, 3] + (2*q0*q2 + 2*q1*q3)*T[2, 3] + (-2*q0*q3 + 2*q1*q2)*T[1, 3] + (2*q0*q3 + 2*q1*q2)*T[3, 3] + (2*q0^2 + 2*q1^2 - 1)*T[0, 3] + (2*q0^2 + 2*q2^2 - 1)*T[4, 3] + (2*q0^2 + 2*q3^2 - 1)*T[8, 3]) + (2*q0^2 + 2*q1^2 - 1)*((-2*q0*q1 + 2*q2*q3)*T[5, 0] + (2*q0*q1 + 2*q2*q3)*T[7, 0] + (-2*q0*q2 + 2*q1*q3)*T[6, 0] + (2*q0*q2 + 2*q1*q3)*T[2, 0] + (-2*q0*q3 + 2*q1*q2)*T[1, 0] + (2*q0*q3 + 2*q1*q2)*T[3, 0] + (2*q0^2 + 2*q1^2 - 1)*T[0, 0] + (2*q0^2 + 2*q2^2 - 1)*T[4, 0] + (2*q0^2 + 2*q3^2 - 1)*T[8, 0]) + (2*q0^2 + 2*q2^2 - 1)*((-2*q0*q1 + 2*q2*q3)*T[5, 4] + (2*q0*q1 + 2*q2*q3)*T[7, 4] + (-2*q0*q2 + 2*q1*q3)*T[6, 4] + (2*q0*q2 + 2*q1*q3)*T[2, 4] + (-2*q0*q3 + 2*q1*q2)*T[1, 4] + (2*q0*q3 + 2*q1*q2)*T[3, 4] + (2*q0^2 + 2*q1^2 - 1)*T[0, 4] + (2*q0^2 + 2*q2^2 - 1)*T[4, 4] + (2*q0^2 + 2*q3^2 - 1)*T[8, 4]) + (2*q0^2 + 2*q3^2 - 1)*((-2*q0*q1 + 2*q2*q3)*T[5, 8] + (2*q0*q1 + 2*q2*q3)*T[7, 8] + (-2*q0*q2 + 2*q1*q3)*T[6, 8] + (2*q0*q2 + 2*q1*q3)*T[2, 8] + (-2*q0*q3 + 2*q1*q2)*T[1, 8] + (2*q0*q3 + 2*q1*q2)*T[3, 8] + (2*q0^2 + 2*q1^2 - 1)*T[0, 8] + (2*q0^2 + 2*q2^2 - 1)*T[4, 8] + (2*q0^2 + 2*q3^2 - 1)*T[8, 8])"
    quartic = quartic.replace("^", "**")
    q = cp.Variable(4)
    for i in range(9):
        for j in range(9):
            quartic = quartic.replace(f'T[{i}, {j}]', str(operator_T[i, j]))
    for i in range(4):
        quartic = quartic.replace(f"q{i}", f"q[{i}]")
    obj = eval(quartic)
    cons = [cp.square(q) == 1]
    prob = cp.Problem(cp.Minimize(obj), cons)

    result = prob.solve()
    return result.value

def minimize_quadratic_equation_on_rotation_qcqp(operator_T):
    x = cp.Variable(9)
    obj = x.T@operator_T@x
    cons = [cp.sum_squares(x[:3])==1, cp.sum_squares(x[3:6])==1, cp.sum_squares(x[6:])==1, x[:3].T@(x[3:6])==0, x[6:].T@x[3:6]==0, x[:3].T@x[6:]==0]
    prob = cp.Problem(cp.Minimize(obj), cons)
    qcqp = QCQP(prob)
    qcqp.suggest(SDR)
    print("SDR lower bound: %.3f" % qcqp.sdr_bound)

    # Attempt to improve the starting point given by the suggest method
    f_cd, v_cd = qcqp.improve(COORD_DESCENT)
    print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
    y=[x.value[i][0] for i in range(9)]
    print(y)
    return np.array(y).reshape((3, 3))


def get_unique_rotations(r, mode, session):
    operator_mode = construct_torque_operator(r, mode)
    solutions, quaternions = solve_linear_in_quartenions_wolfram(operator_mode, session)
    unique_solutions = []
    indices_solutions = []
    for i, solution in enumerate(solutions):
        use_it = True
        for solution_bis in unique_solutions:
            if np.linalg.norm(solution-solution_bis)<1e-3:
                use_it=False
        if use_it:
            unique_solutions.append(solution)
            indices_solutions.append(i)
    return unique_solutions