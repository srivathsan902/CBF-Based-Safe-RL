import os
import yaml
import torch
import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from wrapperEnv import CustomEnv
from stable_baselines3 import SAC
from safety_gymnasium.wrappers.gymnasium_conversion import make_gymnasium_environment


class neuralCBF(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims = [256]):
        super(neuralCBF, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dims

        self.loss_history = []

        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            else:
                setattr(self, f'fc{i+1}', nn.Linear(hidden_dims[i-1], hidden_dim))
        
        self.fc_out = nn.Linear(hidden_dims[-1], 1)
        

        self.gain_factor = 100
    
    def set_action_limits(self, action_limits):
        self.action_limits = action_limits
        self.action_limits = torch.tensor(self.action_limits, dtype=torch.float32)

    
    def set_properties(self, mass = 0.46786522454870777):
        self.mass = mass

    def get_dynamics(self, state, action):
        x, y, v, theta = state[0], state[1], state[2], state[3]
        F, w = action[0], action[1]

        return [v*np.cos(theta), v*np.sin(theta), F/self.mass, w]

    def forward(self, state, action, debug = False):
        if debug:
            print('Forward Pass in neuralCBF')
            print('State:', state)
            print('Action:', action)
            print('State size:', state.size())
            print('Action size:', action.size())
            print('State type:', state.dtype)
            print('Action type:', action.dtype)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        if len(action.size()) == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim = 1)

        for i in range(len(self.hidden_dim)):
            x = getattr(self, f'fc{i+1}')(x)
            x = torch.tanh(x)
        
        x = self.fc_out(x)

        if debug:
            print('Forward Pass in neuralCBF')
            print('State:', state)
            print('Action:', action)
            print('State size:', state.size())
            print('Action size:', action.size())
            print('State type:', state.dtype)
            print('Action type:', action.dtype)
            print('Output:', x)

        return x

    def rectify_action(self, state, action, debug = False):
        if debug:
            print('State:', state)
            print('Action:', action)
        state.requires_grad_()

        rectified_action = cp.Variable(self.action_dim)

        state_np = state.detach().numpy().flatten()
        action_np = action.detach().numpy().flatten()

        def h_and_grad_h(state_np, action_np):
            action_tensor = torch.tensor(action_np, dtype=state.dtype).reshape(1, -1)
            state_tensor = torch.tensor(state_np, dtype=state.dtype, requires_grad=True).reshape(1, -1)
            
            # Compute h
            h_tensor = self.forward(state_tensor, action_tensor)
            
            # Compute gradient of h with respect to state
            grad_h_tensor = torch.autograd.grad(outputs=h_tensor, inputs=state_tensor, grad_outputs=torch.ones_like(h_tensor), create_graph=True)[0]
            
            return h_tensor.item(), grad_h_tensor.detach().numpy().flatten()
        
        def constraint_and_objective(rectified_action_var):
            h_value, grad_h_np = h_and_grad_h(state_np, rectified_action_var.value)
            dynamics = np.array(self.get_dynamics(state_np, rectified_action.value)).flatten()
            constraint_value = grad_h_np @ dynamics + self.gain_factor * h_value
            return constraint_value, cp.norm(rectified_action - action.detach().numpy().flatten(), 2)

        # Formulate the problem with the initial guess
        rectified_action.value = action_np
        constraint_value, objective_value = constraint_and_objective(rectified_action)
        constraint_expr = cp.Constant(constraint_value) >= 0
        constraints = [constraint_expr]

        # Define the objective
        objective = cp.Minimize(objective_value)

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if debug:
            print('Original Action:', action.detach().numpy().flatten())
            print('Rectified Action:', rectified_action.value)
            print('Objective Value:', problem.value)
            print('Constraint Value:', constraint_value)

        if rectified_action.value is None:
            print("Optimization failed. Returning original action.")
            return action
        
        rectified_action = torch.tensor(rectified_action.value, dtype=state.dtype).reshape(1, -1)
        rectified_action = torch.clamp(rectified_action, min=self.action_limits[:,0], max=self.action_limits[:,1])

        return rectified_action
    
    def train(self, X, y, max_epochs = 100):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = criterion = torch.nn.BCEWithLogitsLoss() 
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

        for epoch in tqdm(range(max_epochs)):
            batch_loss = 0
            for batch in dataloader:
                inputs, labels = batch
                optimizer.zero_grad()
                
                states = inputs[:, :self.state_dim]
                actions = inputs[:, self.state_dim:]
                outputs = self.forward(states, actions, debug=False)
                loss = criterion(outputs, labels)
                batch_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
            
            batch_loss /= len(dataloader)
            self.loss_history.append(batch_loss)
    
    def save(self, path):
        hidden_dims = self.hidden_dim
        loss_history = self.loss_history

        torch.save({
            'hidden_dims': hidden_dims,
            'loss_history': loss_history,
            'model_state_dict': self.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.hidden_dim = checkpoint['hidden_dims']
        self.loss_history = checkpoint['loss_history']

def generate_expert_trajectories():
    with open('src/PointCircle1.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')
    render_mode = params['main'].get('render_mode', None)
    model_name = params['main']['model_name']
    params['task'] = 'Circle'
    params['level'] = '1'
    gym_env_name = env_id.split('-')[0] + 'Gymnasium-v0'

    if render_mode == 'None':
        env = make_gymnasium_environment(gym_env_name)
    else:
        env = make_gymnasium_environment(gym_env_name, render_mode=render_mode)
    
    env = CustomEnv(env, params)

    max_episodes = 1000
    model = SAC.load("artifacts/2024/08/11/Run_2/SAC_1000" + '.zip', env)

    X_data, y_data = [], []
    for episode in tqdm(range(max_episodes)):
        state, info = env.reset()
        for step in range(500):
            pos = env.task.agent.pos
            x, y = pos[0], pos[1]
            vel = state[3]
            theta = env.get_theta()
            
            action = model.predict(state, deterministic=True)[0]

            X_data.append([x, y, vel, theta, action[0], action[1]])

            next_state, reward, done, truncated, info = env.step(action)
            if info['cost'] != 0:
                y_data.append(1)
            else:
                y_data.append(0)

            if done or truncated:
                break
            state = next_state
        if episode % 499 == 0 and episode != 0:
            df = pd.DataFrame(X_data, columns=['x', 'y', 'v', 'theta', 'F', 'w'])
            df['Safe'] = y_data
            if not os.path.exists('Expert_Demonstrations'):
                os.makedirs('Expert_Demonstrations')
            df.to_csv(f'Expert_Demonstrations/Expert_Demonstrations_{episode+1}.csv', index=False)
        

if __name__ == '__main__':

    # generate_expert_trajectories()

    NeuralCBF = neuralCBF(4, 2, [64, 128, 128, 64])
    action_limits = np.array([[-1,1], [-1,1]])
    NeuralCBF.set_action_limits(action_limits)
    NeuralCBF.set_properties()
    
    # state = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    # action = torch.tensor([[0.5, 0.6]], dtype=torch.float32)

    # rectified_action = NeuralCBF.rectify_action(state, action, debug = True)
    
    data = pd.read_csv('Expert_Demonstrations/Expert_Demonstrations_1000.csv')
    X = data.drop('Safe', axis=1).values
    y = data['Safe'].values
    y = 1 - y

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # NeuralCBF.train(X, y, max_epochs=100)
    # NeuralCBF.save('Expert_Demonstrations/NeuralCBF.pt')

    NeuralCBF.load('Expert_Demonstrations/NeuralCBF.pt')
    loss_history = NeuralCBF.loss_history

    import matplotlib.pyplot as plt
    # plt.plot(loss_history)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('CBF Training Loss')
    # plt.show()

    states = X[:,:4]
    actions = X[:,4:]
    # print(actions.shape)
    safety_values = y

    # for i in range(500):
    #     print('Example:', i)
    #     state = states[i]
    #     action = actions[i]
    #     safety = safety_values[i]
        
    #     rectified_action = NeuralCBF.rectify_action(state, action, debug = False)
    #     print('Action', action)
    #     print('Rectified Action', rectified_action)
    #     print('CBF Value:', NeuralCBF.forward(state, action, debug=False))
    #     print('Safety Value:', safety)

    env_id = 'SafetyPointCircle1-v0'
    render_mode = 'human'
    model_name = 'SAC'

    params = {
        'main': {
            'env_id': env_id,
            'render_mode': render_mode,
            'model': model_name,
        },
        'train': {
            'save_every': 1000,
            'record_every': 1000,
            'max_steps_per_episode': 500,
        }
    }
    params['task'] = 'Circle'
    params['level'] = '1'

    gym_env_name = env_id.split('-')[0] + 'Gymnasium-v0'

    if render_mode == 'None':
        env = make_gymnasium_environment(gym_env_name)
    else:
        env = make_gymnasium_environment(gym_env_name, render_mode=render_mode)

    env = CustomEnv(env, params, NeuralCBF.rectify_action)
    model = SAC.load("artifacts/2024/08/11/Run_2/SAC_1000" + '.zip', env)

    state, info = env.reset()

    for step in range(500):
        action = [0.5, 0]

        action = NeuralCBF.rectify_action(torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32))
        next_state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break
        state = next_state

    env.close()