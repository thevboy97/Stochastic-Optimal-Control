from casadi import *
import numpy as np

from starter_code.main import lissajous


# This function implements the CEC controller


def cec_controller(cur_state, cur_ref, cur_iter):

    # horizon
    T = 10

    # opti object
    opti = Opti()

    # variables

    # error
    e = opti.variable(3, T+1)
    e[:, 0] = cur_state - cur_ref

    # controls
    u = opti.variable(2, T)

    # parameters
    Q = opti.parameter(2, 2)
    q = opti.parameter(1, 1)
    R = opti.parameter(2, 2)
    gamma = opti.parameter(1, 1)

    # initialize V
    V = 0

    # constraints
    for t in range(T):
        # error
        ref = lissajous(t + cur_iter)
        ref_next = lissajous(t + cur_iter + 1)
        G_tilda = vertcat(horzcat(cos(e[2, t] + ref[2]), 0), horzcat(
            sin(e[2, t] + ref[2]), 0), horzcat(0, 1))
        e[:, t+1] = e[:, t] + G_tilda @ u[:, t] + ref - ref_next

        # space
        x_coord = e[0, t] + ref[0]
        y_coord = e[1, t] + ref[1]
        # theta = theta_tilda[:, t] + alpha
        opti.subject_to(x_coord >= -3)
        opti.subject_to(x_coord <= 3)
        opti.subject_to(y_coord >= -3)
        opti.subject_to(y_coord <= 3)
        opti.subject_to((x_coord + 2)**2 + (y_coord + 2)**2 > 0.5**2)
        opti.subject_to((x_coord - 1)**2 + (y_coord - 2)**2 > 0.5**2)
        # opti.subject_to(theta >= -pi)
        # opti.subject_to(theta < pi)

        # equation
        V += gamma**t * (e[:2, t].T @ Q @ e[:2, t] +
                         q * (1 - cos(e[2, t]))**2 + u[:, t].T @ R @ u[:, t])
    # add terminal cost
    V += e[:2, T].T @ Q @ e[:2, T] + \
        q * (1 - cos(e[2, T]))**2

    # controls
    vel = u[0]
    omega = u[1]
    opti.subject_to(vel >= 0)
    opti.subject_to(vel <= 1)
    opti.subject_to(omega >= -1)
    opti.subject_to(omega <= 1)

    # set values
    opti.set_value(Q, 10*np.identity(2))
    opti.set_value(q, 100)
    opti.set_value(R, 20*np.identity(2))
    opti.set_value(gamma, 0.9)

    # minimize
    opti.minimize(V)

    # solver
    p_opts = dict(print_time=False, verbose=False)
    s_opts = dict(print_level=0)
    opti.solver('ipopt', p_opts, s_opts)

    # solve
    sol = opti.solve()

    # extract v and w
    v = sol.value(u[0, 0])
    w = sol.value(u[1, 0])

    return [v, w]
