import numpy as np
import sympy as sp
import time
import safety_gymnasium

import cvxpy as cp

def get_theta(state):
    '''
    Get the angle of the agent from the state
    '''
    theta = np.arctan2(state[9], state[10])*180/np.pi
    theta = (theta + 180) % 360

    return theta*np.pi/180

def get_velocity(local_v_x, local_v_y, theta):
    '''
    Get the global x and y components of the velocity
    '''
    v_x = local_v_x*np.cos(theta) - local_v_y*np.sin(theta)
    v_y = local_v_x*np.sin(theta) + local_v_y*np.cos(theta)

    return v_x, v_y

def CBF(state, pos, action, low, high, debug = False):
    '''
    Add as many constraints as needed here
    
    '''
    # ********* Define the parameters of the system *********
    d_thresh = 0.1
    d_thresh_vel = 0.1
    r = 0.1
    max_lidar_distance = 6
    mass = 0.46786522454870777

    x, y, v, theta = sp.symbols('x y v theta')

    # ********* Define the action variables of the system *********
    Force = sp.symbols('F')
    omega = sp.symbols('w')

    control_input = sp.Matrix([Force, omega])

    # ********* Define the dynamics of the system *********
    f = sp.Matrix([v*sp.cos(theta), v*sp.sin(theta), 0, 0])
    g = sp.Matrix([[0, 0], [0, 0], [1/mass, 0], [0, 1]])

    X_dot = f + g*control_input

    theta_val = get_theta(state)
    v_val = state[3]

    # ********* Extract lidar values from the state *********
    lidar_values = state[-16:]  # Assuming the last 16 values in the state are lidar values
    lidar_angles = np.linspace(0, 2*np.pi, len(lidar_values))  # Assuming 180 degree lidar scan

    # ********* Define the Control Barrier Functions h *********
    cbf_constraints = []
    k1 = 100  # CBF gain
    k2 = 100
    u = cp.Variable(2)

    x_star = x + r*sp.cos(theta)
    y_star = y + r*sp.sin(theta)

    for lidar_value, lidar_angle in zip(lidar_values, lidar_angles):
        if lidar_value <= 0.90:
            continue
        
        lidar_distance = (1-lidar_value)*max_lidar_distance
        x_obs = pos[0] + lidar_distance*sp.cos(theta_val + lidar_angle)
        y_obs = pos[1] + lidar_distance*sp.sin(theta_val + lidar_angle)

        h_position = (x_star - x_obs)**2 + (y_star - y_obs)**2 - d_thresh**2

        rel_x_obs = x_obs - x
        rel_y_obs = y_obs - y
        if v_val*sp.cos(theta_val)*(x_obs - pos[0]) + v_val*sp.sin(theta_val)*(x_obs - pos[1]) < 0:
            h_velocity = 0.01 - (v*sp.cos(theta)*(rel_x_obs + d_thresh_vel*sp.cos(theta_val)) + v*sp.sin(theta)*(rel_y_obs + d_thresh_vel*sp.sin(theta_val)))
        elif v_val*sp.cos(theta_val)*(x_obs - pos[0]) + v_val*sp.sin(theta_val)*(x_obs - pos[1]) > 0:
            h_velocity = 0.01 - (v*sp.cos(theta)*(rel_x_obs - d_thresh_vel*sp.cos(theta_val)) + v*sp.sin(theta)*(rel_y_obs - d_thresh_vel*sp.sin(theta_val)))
        else:
            h_velocity = 0.01

        grad_h_position = sp.Matrix([sp.diff(h_position, var) for var in (x, y, v, theta)])
        grad_h_velocity = sp.Matrix([sp.diff(h_velocity, var) for var in (x, y, v, theta)])

        position_constraint = grad_h_position.dot(X_dot) + k1*h_position
        velocity_constraint = grad_h_velocity.dot(X_dot) + k2*h_velocity

        cbf_constraints.append(position_constraint)
        cbf_constraints.append(velocity_constraint)

    # Convert symbolic expressions to numeric functions
    constraint_funcs = [sp.lambdify((x, y, v, theta, Force, omega), cbf, 'numpy') for cbf in cbf_constraints]

    # Define CVXPY variables
    u = cp.Variable(2)
    
    # Define the objective function (minimize deviation from original action)
    objective = cp.Minimize(cp.sum_squares(u - action))
    
    # Define the constraints
    constraints = [
        cbf_func(pos[0], pos[1], v_val, theta_val, u[0], u[1]) >= 0 for cbf_func in constraint_funcs
    ]

    if debug:
        print('***************Before solving the optimization problem*****************')
        for i, cbf_func in enumerate(constraint_funcs):
            print(f'Constraint {i+1}', cbf_func(pos[0], pos[1], v_val, theta_val, action[0], action[1]))
        print('\n')

    # Solve the optimization problem

    all_constraints_satisfied_flag = True
    for func in constraint_funcs:
        if func(pos[0], pos[1], v_val, theta_val, action[0], action[1]) < 0:
            all_constraints_satisfied_flag = False
            break
    
    optimizer_used = False
    if all_constraints_satisfied_flag:
        if debug:
            print('All constraints satisfied')
        return action, optimizer_used
    else:
        optimizer_used = True

    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if u.value is not None:
        safe_action = u.value
        safe_action = np.clip(safe_action, low, high)

        tolerance = 1e-4
        candidate_actions = [safe_action]
        num_random_samples = 20  # Number of random samples to check for equivalent solutions
        for _ in range(num_random_samples):
            random_action = np.random.uniform(low, high, 2)
            if prob.constraints[0].dual_value is not None:
                random_constraints = [
                    cbf_func(pos[0], pos[1], v_val, theta_val, random_action[0], random_action[1]) >= 0 for cbf_func in constraint_funcs
                ]
                if np.all(random_constraints):
                    random_objective = np.sum((random_action - action) ** 2)
                    if np.abs(random_objective - prob.value) < tolerance:
                        candidate_actions.append(random_action)

        if len(candidate_actions) > 1:
            safe_action = candidate_actions[np.random.randint(len(candidate_actions))]

        if debug:
            print('***************After solving the optimization problem*****************')
            print('Unsafe action  ', action, '  Safe action found  ', safe_action)
            for i, cbf_func in enumerate(constraint_funcs):
                print(f'Constraint {i+1}', cbf_func(pos[0], pos[1], v_val, theta_val, safe_action[0], safe_action[1]))
            print('\n')
        return safe_action, optimizer_used
    else:
        if debug:
            print('No safe action found')
        return action, optimizer_used
