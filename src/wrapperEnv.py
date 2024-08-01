import os
import yaml
import wandb
import gymnasium
import numpy as np
from CBF import CBF
from stable_baselines3 import SAC
from src.callback import CustomCallback
from safety_gymnasium.wrappers.gymnasium_conversion import make_gymnasium_environment


class CustomEnv(gymnasium.Wrapper):
    def __init__(self, env, params):
        super().__init__(env)
        self.env = env
        self.params = params
        self.max_steps = params['train'].get('max_steps_per_episode', 250)
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.state = self.env.reset()[0]
        self.pos = self.env.task.agent.pos
        self.vel = self.get_velocities()
        self.num_steps = 0
    
    def reset(self, **kwargs):
        if 'seed' in kwargs:
            kwargs.pop('seed')
        self.state, info = self.env.reset()
        self.pos = self.env.task.agent.pos
        self.vel = self.get_velocities()
        info['position'] = self.pos
        info['velocity'] = self.vel
        return self.state, info

    def step(self, action):

        safe_action, cbf_optimizer_used = CBF(np.array(self.state), self.pos, action, self.action_low, self.action_high)
        next_state, reward, done, truncated, info = self.env.step(safe_action)
        self.num_steps += 1

        if self.num_steps >= self.max_steps:
            truncated = True
            self.num_steps = 0

        info['cbf_optimizer_used'] = cbf_optimizer_used
        info['reward'] = reward
        info['episode_end'] = done or truncated
        info['action'] = safe_action
        info['velocity'] = self.vel
        info['position'] = self.pos

        self.state = next_state
        self.pos = self.env.task.agent.pos
        self.vel = self.get_velocities()

        return next_state, reward, done, truncated, info
    
    def close(self):
        self.env.close()

    def get_theta(self):
        theta = np.arctan2(self.state[9], self.state[10])*180/np.pi
        theta = (theta + 180) % 360
        return theta*np.pi/180
    
    def get_velocities(self):
        local_v_x = self.state[3]
        local_v_y = self.state[4]
        theta = self.get_theta()
        v_x = local_v_x*np.cos(theta) - local_v_y*np.sin(theta)
        v_y = local_v_x*np.sin(theta) + local_v_y*np.cos(theta)

        return (v_x, v_y)


if __name__ == '__main__':
    with open('CBF_DDPG/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project="Stable Baselines")
    env = make_gymnasium_environment('SafetyPointCircle1Gymnasium-v0', render_mode='human')

    env = CustomEnv(env)

    model = SAC("MlpPolicy", env, verbose=0)

    callback = CustomCallback(params)

    model.learn(total_timesteps=1500, callback=callback)

    wandb.finish()