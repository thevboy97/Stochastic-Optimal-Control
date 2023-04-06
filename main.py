from time import time
import numpy as np
from utils import visualize
from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5  # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k


def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller


def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0]) **
                    2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v, w]

# This function implements the CEC controller


def cec_controller(cur_state, cur_ref, cur_iter):

    # horizon
    T = 10

    # time step
    delta = time_step

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
        G_tilda = vertcat(horzcat(delta * cos(e[2, t] + ref[2]), 0), horzcat(
            delta * sin(e[2, t] + ref[2]), 0), horzcat(0, delta))
        e[:, t+1] = e[:, t] + G_tilda @ u[:, t] + ref - ref_next

        # space
        x_coord = e[0, t] + ref[0]
        y_coord = e[1, t] + ref[1]
        opti.subject_to(x_coord >= -3)
        opti.subject_to(x_coord <= 3)
        opti.subject_to(y_coord >= -3)
        opti.subject_to(y_coord <= 3)
        opti.subject_to((x_coord + 2)**2 + (y_coord + 2)**2 > 0.5**2)
        opti.subject_to((x_coord - 1)**2 + (y_coord - 2)**2 > 0.5**2)

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
    opti.set_value(Q, 20*np.identity(2))
    opti.set_value(q, 40)
    opti.set_value(R, 2*np.identity(2))
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

# This function implement the car dynamics


def car_next_state(time_step, cur_state, control, noise=True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04  # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()


if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = simple_controller(cur_state, cur_ref)
        control = cec_controller(cur_state, cur_ref, cur_iter)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(
            time_step, cur_state, control, noise=False)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        # error = error + np.linalg.norm(cur_state - cur_ref)
        error += np.linalg.norm(cur_state[:2] - cur_ref[:2])
        angle_diff = np.abs((cur_state[2] - cur_ref[2]) % (2*np.pi))
        error += min(angle_diff, 2*np.pi - angle_diff)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)
