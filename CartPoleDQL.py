# Importing the classes 
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F 
import torch.optim as optim
import gym
from collections import namedtuple, deque

# Defining the Neural Network 
class Network(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.states = state_shape
        self.actions = action_shape

        # defining our layers
        self.fc1 = nn.Linear(in_features = self.states, out_features = 24)
        self.fc2 = nn.Linear(in_features = 24, out_features = 12)
        self.out = nn.Linear(in_features = 12, out_features = self.actions)

    
    # forward method 
    def forward(self, t): 
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t 

# Defining the experience tuple 
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'terminal'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Experience(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        experience = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*experience))
        return batch

    def __len__(self):
        return len(self.memory)


def train_model(policy_net, target_net, optimizer, batch):
    # Extracting states, next_states, actions, rewards, and done variable in current batch
    states = torch.stack(batch.state) 
    next_states = torch.stack(batch.next_state) # 
    actions = torch.Tensor(batch.action).float().to(device) 
    rewards = torch.Tensor(batch.reward).to(device)
    terminals = torch.Tensor(batch.terminal).to(device)

    # Predicting the current and next_state-action pair q-values 
    policy_qs = policy_net(states).squeeze(1) 
    target_qs = target_net(next_states).squeeze(1)  
    policy_qs = torch.sum(policy_qs.mul(actions), dim=1)  

    # Applying the Bellman Equation: R(s, a) + max_a' Q(s', a')
    target_qs = rewards + terminals * discount_factor * target_qs.max(1)[0]
    
    # Hubber-Loss Function is applied 
    loss = F.mse_loss(policy_qs, target_qs.detach()) 

    # Calculating Gradients + Back Propagation and Gradient Descent
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    return loss

def get_action(input, model, epsilon, env):
    # Epsilon-Greedy Strategy 
    if np.random.rand() <= epsilon:
        action = env.action_space.sample()
        return action

    else:
        qvalue = model.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]


# defining the environment
env = gym.make('CartPole-v1')

# Defining the hyper-paramaters     
epsilon = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exp_decay = 0.001
discount_factor = 0.99
replay_memory = Memory(100000)
learning_rate = 0.001
batch_size = 64
steps_until_traning = 0

# Setting a device for pytorch (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining our neural network
policy_net = Network(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = Network(env.observation_space.shape[0], env.action_space.n).to(device)

# Setting the same parameters as the policy net for our target net 
target_net.load_state_dict(policy_net.state_dict())

# Using the Adam Optimizer
optimizer = optim.Adam(params = policy_net.parameters(), lr = learning_rate)

# Setting both the policy_net and target_net in training mode
policy_net.train()
target_net.train()

# Iterating through episodes 
for episode in range(1000):
    state = torch.Tensor(env.reset()).unsqueeze(0).to(device)
    total_rewards_per_episode = 0
    done = False

    #Iterating through time_steps in current episode 
    while not done:
        # if True:
        #     env.render()

        steps_until_traning+=1
        action = get_action(state, policy_net, epsilon, env)
        next_state, reward, done, info = env.step(action)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

        # terminal = done variable; if done is True, a 0 will be returned. Otherwise, a 1 will be returned
        terminal = 0 if done else 1

        # turning the action into a vector with shape (2,)
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1 

        # pusing the experience into the replay memory list 
        replay_memory.push(state, action_one_hot, reward, next_state, terminal)
        # for every 4 time_steps, the networks will train

        if steps_until_traning % 4 == 0 or done:
            if len(replay_memory) > batch_size:
                batch = replay_memory.sample(batch_size)
                loss = train_model(policy_net, target_net, optimizer, batch)

        state = next_state
        total_rewards_per_episode += reward
        
        if done: 
            print(f'Total training rewards: {total_rewards_per_episode} after n steps = {episode} with eps: {epsilon} with action {action}')
            if steps_until_traning >= 100:
                print('Copying main network weights to the target network weights')
                target_net.load_state_dict(policy_net.state_dict())
                steps_until_traning = 0
            break
        
    # Updating the current exploration rate (epislon) by applying the decay rate 
    epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exp_decay * episode)
  

