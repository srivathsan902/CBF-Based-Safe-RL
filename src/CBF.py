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
    # print(state)
    # state = state.squeeze(0).cpu().numpy()
    d = 1.125
    d = 1.1
    d_thresh = 0.5
    r = 0.1
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

    # ********* Define the Control Barrier Functions h *********
    x_star = x + r*sp.cos(theta)
    h1 = (d**2 - x_star**2)
    
    if debug:
        print('x : ', pos[0], ' v_x', v_val*np.cos(theta_val))

    # if pos[0] < 0 and v_val*np.cos(theta_val) < 0:
    #     h2 = 0.01 - (x+d_thresh)*v*sp.cos(theta)
    # elif pos[0] > 0 and v_val*np.cos(theta_val) > 0:
    #     h2 = 0.01 - (x-d_thresh)*v*sp.cos(theta)
    # else:
    #     h2 = 0.01
    
    if v_val*np.cos(theta_val) < 0:
        h2 = 0.01 - (x+d_thresh)*v*sp.cos(theta)
    elif v_val*np.cos(theta_val) > 0:
        h2 = 0.01 - (x-d_thresh)*v*sp.cos(theta)
    else:
        h2 = 0.01
    

    # ********* Formulate the constraints *********
    grad_h1 = sp.Matrix([sp.diff(h1, var) for var in (x, y, v, theta)])
    grad_h2 = sp.Matrix([sp.diff(h2, var) for var in (x, y, v, theta)])
    
    k1 = 500
    k2 = 100
    
    constraint_1 = grad_h1.dot(X_dot) + k1*(h1)
    constraint_2 = grad_h2.dot(X_dot) + k2*(h2)
    

    # Convert symbolic expressions to numeric functions
    constraint_1_func = sp.lambdify((x, y, v, theta, Force, omega), constraint_1, 'numpy')
    constraint_2_func = sp.lambdify((x, y, v, theta, Force, omega), constraint_2, 'numpy')
    
    constraint_funcs = [constraint_1_func, constraint_2_func]
    # Define CVXPY variables
    u = cp.Variable(2)
    
    # Define the objective function (minimize deviation from original action)
    objective = cp.Minimize(cp.sum_squares(u - action))
    
    # Define the constraints
    constraints = [
        constraint_1_func(pos[0], pos[1], v_val, theta_val, u[0], u[1]) >= 0,
        constraint_2_func(pos[0], pos[1], v_val, theta_val, u[0], u[1]) >= 0,
        
    ]

    if debug:
        print('***************Before solving the optimization problem*****************')

        print('Constraint 1', constraint_1_func(pos[0], pos[1], v_val, theta_val, action[0], action[1]))
        print('Constraint 2', constraint_2_func(pos[0], pos[1], v_val, theta_val, action[0], action[1]))

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
            if prob.constraints[0].dual_value is not None and prob.constraints[1].dual_value is not None:
                random_constraints = [
                    constraint_1_func(pos[0], pos[1], v_val, theta_val, random_action[0], random_action[1]) >= 0,
                    constraint_2_func(pos[0], pos[1], v_val, theta_val, random_action[0], random_action[1]) >= 0,
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
            print('Constraint 1', constraint_1_func(pos[0], pos[1], v_val, theta_val, safe_action[0], safe_action[1]))
            print('Constraint 2', constraint_2_func(pos[0], pos[1], v_val, theta_val, safe_action[0], safe_action[1]))
            
            print('\n')
        return safe_action, optimizer_used
    else:
        if debug:
            print('No safe action found')
        return action, optimizer_used


if __name__ == "__main__":
    env = safety_gymnasium.make('SafetyPointCircle1-v0', render_mode='human')
    state, info = env.reset()
    
    low = env.action_space.low
    high = env.action_space.high
    cnt = 0
    action = env.action_space.sample()
    # action = [0.5,0.5]
    for i in range(500):
        # action = env.action_space.sample()
        pos = env.task.agent.pos
        vel = env.task.agent.vel
        action = [1,0.5]
        safe_action = CBF(state, pos, action, low, high)
        next_state, reward, cost, done, truncated, _ = env.step(safe_action)

        time.sleep(0.05)
        if done or truncated:
            break
        state = next_state