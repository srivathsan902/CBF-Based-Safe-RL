import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from wrapperEnv import CustomEnv
from safety_gymnasium.wrappers.gymnasium_conversion import make_gymnasium_environment
from dotenv import load_dotenv
import yaml
import os
import time

load_dotenv()

with open('src/params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

params_file_name = params['agent'] + params['task'] + params['level'] + '.yaml'

with open(os.path.join('src', params_file_name), 'r') as f:
    params = yaml.safe_load(f)

params['task'] = 'Goal'
env_id = params['main'].get('env_id', 'SafetyPointGoal1-v0')
render_mode = params['main'].get('render_mode', None)
model_name = params['main']['model_name']

if render_mode == 'None':
    gym_env_name = env_id.split('-')[0] + 'Gymnasium-v0'
    env = make_gymnasium_environment(gym_env_name)
else:
    gym_env_name = env_id.split('-')[0] + 'Gymnasium-v0'
    env = make_gymnasium_environment(gym_env_name, render_mode=render_mode)
from CBF_Lidar_Based import CBF
import matplotlib.pyplot as plt
# env = CustomEnv(env, params, CBF = None)
env = CustomEnv(env, params, CBF = CBF)
action = env.action_space.sample()
fig, ax = plt.subplots()
debug = False
for episode in range(10):
    for i in range(500):
        action = [0.5,0]
        next_state, reward, done, truncated, info = env.step(action)
        x, y, _ = env.task.agent.pos
        goal_pos = env.task.goal.pos
        hazard_pos = env.task.hazards.pos
        print(len(hazard_pos), hazard_pos[0])
        theta = np.arctan2(next_state[9], next_state[10])*180/np.pi
        theta = (theta + 180) % 360
        theta = theta*np.pi/180
        if debug:
            plt.plot(x, y, 'ro-')
            plt.plot(x+0.5*np.cos(theta), y+0.5*np.sin(theta), 'bo-')
            lidar_x = []
            lidar_y = []
            print(next_state[-32:-16])
            for lidar_value, lidar_angle in zip(next_state[-32:-16], np.linspace(0, 2*np.pi, 16, endpoint=False)):
                lidar_distance = (1-lidar_value)*3
                x_lid = x + lidar_distance*np.cos(theta+lidar_angle)
                y_lid = y + lidar_distance*np.sin(theta+lidar_angle)
                lidar_x.append(x_lid)
                lidar_y.append(y_lid)
            # pLot first point alone in green
            plt.plot(lidar_x, lidar_y, 'go-')
            # Plot the rest of the points in pink
            # plt.plot(lidar_x[1:], lidar_y[1:], 'mo-')

            lidar_x = []
            lidar_y = []
            print(next_state[-16:])
            for lidar_value, lidar_angle in zip(next_state[-16:], np.linspace(0, 2*np.pi, 16, endpoint=False)):
                lidar_distance = (1-lidar_value)*3
                x_lid = x + lidar_distance*np.cos(theta+lidar_angle)
                y_lid = y + lidar_distance*np.sin(theta+lidar_angle)
                lidar_x.append(x_lid)
                lidar_y.append(y_lid)
            plt.plot(lidar_x, lidar_y, 'yo-')
            # Plot the rest of the points in pink
            # plt.plot(lidar_x[1:], lidar_y[1:], 'bo-')


            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.title(f'Step {i+1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax.set_aspect('equal', 'box')


            # print(next_state.shape)
            # print(next_state)
            plt.show()
        # time.sleep(2)
        if done or truncated:
            obs = env.reset()
env.close()

