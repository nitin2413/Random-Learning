import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

### Hyperparameter ###

env_name = "CartPole-v1"
episodes = 500
max_step_per_episode = 500

gamma = 0.99 # discount factor -> 0.99 cares more about future reward
alpha = 1e-3 # learning rate

# 1 - epsilon gives weather we want to explore or exploit a state
# explore -> randomly select next state , action
# exploit -> select best next state , action
# With probability ε, choose a random action
# With probability 1 − ε, choose the best known action
epsilon_start = 1.0 # -> explore
epsilon_end = 0.01 # -> exploit
epsilon_decay = 300 # -> at beginning epsilon 1 as episode increase epsilon decrease

hidden_size = 128

device = torch.device("mps" if torch.mps.is_available() else "cpu")

###  Q-Network ###

class QNetwork(nn.Module):
    def __init__(self,state_dim , action_dim , hidden_size = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,action_dim) # Q(s,a) for each action
        )

    def forward(self,x):
        return self.net(x)

###  ε-greedy policy  ###

def selectaction(q_net, state , epsilon):
    if random.random() < epsilon: # if random generated number less than epsilon explore
        return env.action_space.sample()
    else: # convert state into tensor to use neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():# select the action,state which give max q_value
            q_values = q_net(state_tensor).to(device)
        return q_values.argmax(dim = 1).item()

### Training LOOP ###

env = gymnasium.make(env_name) # make environment (state, motive of its)
state_dim = env.observation_space.shape[0]# tells how many inputs the state has. this return vector which are converted to scaler output using shape[0] over here 4
action_dim = env.action_space.n # tells number of action we can perform over here (up , down , left , right)

q_net = QNetwork(state_dim,action_dim,hidden_size).to(device)
optimizer = optim.Adam(q_net.parameters(), lr=alpha)
loss_Function= nn.MSELoss()

episodes_reward = []# collect reward of an episode

for episode in range(episodes):
    reset_output = env.reset()# reset the environment for every episode as it should start from starting
    if isinstance(reset_output,tuple):
        state , _ = reset_output # new version return state , info(return starting state)
    else:
        state = reset_output # old version return only state
    # formula for epsilon to get decay slowly
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              np.exp(-1.0 * episode / epsilon_decay)

    total_reward = 0.0 # reward for this specific episode
    # this loop is for actions which will be perform in that episode
    for i in range(max_step_per_episode):

        action = selectaction(q_net,state,epsilon) # select action
        step_output = env.step(action)
        # after env.step execute the action it return 5 value
        # next state , reward , terminated
        next_state , reward , terminated, truncate, info = step_output
        done = terminated or truncate

        total_reward += reward

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        q_values = q_net(state_tensor).to(device) #[1, action ]
        # example q_values = [2.42,3.56] action [0,1] if action 0 is choose by selectaction then
        # q_sa = q_value[0,0] -> q_sa = 2.42
        q_sa = q_values[0,action] # select q_value of chosen action

        # to avoid mixing of gradiant in back propogation
        with torch.no_grad(): #max_a' Q(s', a')

            next_q_value = q_net(next_state_tensor).to(device) # [1, action_dim]
            max_next_q_value = next_q_value.max(dim =1)[0].item()# selection of maximum of next state q_value

            if done :
                # when no future exist after this state. (Episode is finished)
                td_target = torch.tensor(reward, dtype=torch.float32, device=device)
            else:
            # This is theBellman
            # equation:Q(s, a) = r + gamma (max Q(s', a'))
            # calculating future reward
                td_target = torch.tensor(reward + gamma * max_next_q_value,
                                         dtype=torch.float32, device=device)
            # if done:
            #     td_target = reward
            # else:
            #     td_target = reward + gamma * max_next_q

        loss = loss_Function(q_sa , td_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
            # 	done = False → episode is still running
            # 	done = True → episode has ended
        if done:
             break
    episodes_reward.append(total_reward)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episodes_reward[-10:])
        print(f"Episode {episode + 1}/{episodes} | "f"Avg reward (last 10): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

env.close()