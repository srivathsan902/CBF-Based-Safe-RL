import os
import sys
import time
import yaml
import wandb
import shutil
import numpy as np

from dotenv import load_dotenv
from safety_gymnasium.wrappers.gymnasium_conversion import make_gymnasium_environment

from wrapperEnv import CustomEnv
from stable_baselines3 import SAC
from src.callback import CustomCallback


load_dotenv()

artifacts_folder = 'artifacts'
# os.environ['WANDB_MODE'] = 'offline'

def main(dir_name, params):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')
    render_mode = params['main'].get('render_mode', None)
    model_name = params['main']['model_name']

    if render_mode == 'None':
        env = make_gymnasium_environment('SafetyPointCircle1Gymnasium-v0')
    else:
        env = make_gymnasium_environment('SafetyPointCircle1Gymnasium-v0', render_mode=render_mode)

    env = CustomEnv(env, params)

    wandb_enabled = params['base']['wandb_enabled']

    if wandb_enabled:
        try:
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)

        except Exception as e:
            print(f"Error occurred while logging into wandb: {e}")
            sys.exit(1)

        run_name = dir_name.replace('/','-').replace('\\','-').replace('artifacts-', "")
        config = {
            'env_id': env_id,
            'render_mode': render_mode,
            'model': model_name,
            'run_name': f'{env_id}-{run_name}',
            'save_every': params['train']['save_every'],
            'record_every': params['train']['record_every'],
        }


        wandb.init(project='Stable Baselines', name = f'{model_name}-{env_id}-{run_name}', config = config)
    
    if params['main'].get('update', False):
        if wandb_enabled:
            run_name = params['main']['update_from'].replace('/','-').replace('\\','-').replace('artifacts-', "")
            start_episode_num = int(run_name.split('-')[-1].split('_')[-1])
            run_name = '-'.join(run_name.split('-')[:-1]) + '-' +  str(start_episode_num)
            artifact_name = model_name + '_' + str(start_episode_num)
            artifacts = wandb.use_artifact(f'{artifact_name}:{run_name}', type="model")
            artifacts_dir = artifacts.download(root = dir_name)
            model = SAC.load(os.path.join(dir_name, artifact_name+'.zip'), env)


            data_artifacts = ['episode_rewards', 'episode_costs', 'episode_percent_safe_actions', 'episode_safety_calls']
            download_name = run_name.split('-')[:-1]
            download_name = '-'.join(download_name)

            for artifact in data_artifacts:
                artifacts = wandb.use_artifact(f'{artifact}:{download_name}', type="data")
                artifacts_dir = artifacts.download(root = dir_name)
            
            all_rewards = np.load(os.path.join(dir_name, 'episode_rewards.npy')).tolist()
            all_costs = np.load(os.path.join(dir_name, 'episode_costs.npy')).tolist()
            all_corrective_actions = np.load(os.path.join(dir_name, 'episode_safety_calls.npy')).tolist()
            all_safe_actions = np.load(os.path.join(dir_name, 'episode_percent_safe_actions.npy')).tolist()
                
        else:
            model = SAC.load(params['main']['update_from']+'.zip', env)
            run_name = params['main']['update_from'].replace('/','-').replace('\\','-').replace('artifacts-', "")
            update_dir = os.path.dirname(params['main']['update_from'])
            start_episode_num = int(run_name.split('-')[-1])
            model.save(os.path.join(dir_name, model_name + f'_{start_episode_num}.zip'))

            if not os.path.exists(os.path.join(update_dir, 'episode_rewards.npy')):
                all_rewards = np.load(os.path.join(update_dir, 'episode_rewards.npy')).tolist()
            else:
                all_rewards = [0]*start_episode_num
            if not os.path.exists(os.path.join(update_dir, 'episode_costs.npy')):
                all_costs = np.load(os.path.join(update_dir, 'episode_costs.npy')).tolist()
            else:
                all_costs = [0]*start_episode_num
            if not os.path.exists(os.path.join(update_dir, 'episode_safety_calls.npy')):
                all_corrective_actions = np.load(os.path.join(update_dir, 'episode_safety_calls.npy')).tolist()
            else:
                all_corrective_actions = [0]*start_episode_num
            if not os.path.exists(os.path.join(update_dir, 'episode_percent_safe_actions.npy')):
                all_safe_actions = np.load(os.path.join(update_dir, 'episode_percent_safe_actions.npy')).tolist()
            else:
                all_safe_actions = [100]*start_episode_num
            
        data = { 
                'rewards': all_rewards,
                'costs': all_costs,
                'corrective_actions': all_corrective_actions,
                'safe_actions': all_safe_actions,
            }
        
        callback = CustomCallback(params, dir_name)
        callback.load_previous_run(data, start_episode_num)


    else:
        model = SAC("MlpPolicy", env, verbose=0, buffer_size=100000)
        callback = CustomCallback(params, dir_name)

    total_timesteps = params['train']['total_num_steps']

    model.learn(total_timesteps=total_timesteps, callback=[callback], progress_bar=True, log_interval=1e9)
    
    env.close()
    if wandb_enabled:
        wandb.finish()


if __name__ == '__main__':

    with open('src/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    '''
    Create the dir name based on the current time: dd_mm_yyyy_hh_mm_ss
    artifacts/yyyy/mm/dd/hh_mm should be the structure
    '''
    dir_name = os.path.join(artifacts_folder,
                        time.strftime('%Y'),
                        time.strftime('%m'),
                        time.strftime('%d'))
    
    run = 1
    while os.path.exists(os.path.join(dir_name, f'Run_{run}')):
        run += 1
    dir_name = os.path.join(dir_name, f'Run_{run}')

    main(dir_name, params)
    


