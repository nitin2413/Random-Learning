import random
import gymnasium
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

#### hyperparameter ####

device = "mps" if torch.mps.is_available() else "cpu"

episodes = 400
max_step_per_episode = 500
batch_size = 64
seed = 42

gamma = 0.99
alpha = 1e-4 # slow learning rate

start_eps = 1.0
end_eps = 0.05
eps_decay = 1500

# max capacity which can be store in the buffer
buffer_capacity = 20000
# in dqn we have 2 network, online and target network
# online network update q_value after every transition , change after every step  → causing instability.
# target network update q_value after 1000 transition, slowly updated frozen copy(replay buffer stored value)
# Steps 1–999     → target_net stays frozen
# Step 1000       → target_net = online_net
# Steps 1001–1999 → target_net stays frozen
# Step 2000       → update again
# 200 as cart pole is small and fast environment
target_update_freq = 200 # after 200 transition update target network
# Before training begins, your agent must first collect 1000 random transitions.
# Why? Because training with too little data makes Q-network unstable.
min_replay_size = 1000

use_double = True # True = Double DQN target; False = standard DQN(q-learning)
use_dueling = True # True = use dueling architecture; False = vanilla head

####  Replay Buffer ####

class ReplayBuffer:
    def __init__(self,capacity,state_shape,dtype=np.float32):
        self.capacity = int(capacity)
        self.state_shape = state_shape # dimension of state
        self.ptr = 0 # value adding pointer
        self.size = 0 # tell us how much capacity used

        self.state = np.zeros((self.capacity,self.state_shape), dtype=dtype) # dtype = float32 optimize for pytorch neural net
        self.next_state = np.zeros((self.capacity,self.state_shape), dtype=dtype)
        self.action = np.zeros((self.capacity,), dtype=np.int64 ) # int64 as it does not return decimal value
        self.reward = np.zeros((self.capacity,), dtype= np.float32) # dt ->as int return decimal value 0.5
        self.dones = np.zeros((self.capacity,),dtype=np.uint8) #dt -> as it can only return 0, 1 safe memory

    def add(self, state , next_state , action , reward , dones):
        # over here ptr act as an index
        # if ptr get > capacity it starts replacing old value with new
        # example ptr 1001 and capacity 1000 then ptr replace ptr[1] with ptr[1001]
        # we not give infinite memory to buffer as its not practical (can use 2 gb of ram)
        # Replay memory is used to approximate the experience distribution of the current policy.
        # Old experience = from old policy , New experience = from improved policy
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.capacity # suppose max cap is 1000 after 1000 ptr -> 1000 + 1 % 1000 == 1 so the new result will save at first position
        self.size = min(self.size+1 , self.capacity) # max size == capacity

    def sample(self,batch_size):
        # idx select random value from buffer, selection == batch_size , random selection as Sequential environment data is correlated.
        idx = np.random.randint(0 , self.size, size = batch_size)
        # values are stored in Batch upto batch size
        Batch = dict(
            state = self.state[idx],
            next_state = self.next_state[idx],
            action = self.action[idx],
            reward = self.reward[idx],
            dones = self.dones[idx]
        )
        return Batch

    # purpose to show current number of stored transitions
    def __len__(self):
        return self.size

####   Q-Network   ####

class QNetwork(nn.Module):
    def __init__(self, state_dim , n_action , hidden_layer = 64 , dueling = False):
        super().__init__()
        self.dueling = dueling
        # common feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU()
        )

        # In many states, the choice of action does not matter much,
        # but the value of being in that state matters a lot.
        # Dueling DQN does NOT skip value computation;
        # it separates state value and action advantage so the agent can learn which states matter even when actions do not.
        if dueling:
            # value stream
            # How good is this state overall?
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_layer,hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer,1)
            )
            # advantage stream
            # How much better is this action compared to others in this state?
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_layer,hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer,n_action)
            )
        else:
            # normal head outputs Q-values for all actions
            self.head = nn.Sequential(
                nn.Linear(hidden_layer,hidden_layer),
                nn.ReLU(),
                nn.Linear(hidden_layer,n_action)
            )

    def forward(self,x):
        z = self.feature(x)
        if self.dueling:
            v = self.value_stream(z) # [B,1]
            a = self.advantage_stream(z) #[B ,n_action]
            # combine: subtract mean advantage for identifiability
            # we take mean of actions
            a_mean= a.mean(dim = 1, keepdim=True)
            q_value = v  + (a - a_mean)
            return q_value
        else:
            return self.head(z)

def set_seed(env , seed = seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:# Some environments might have a separate random number generator for selecting actions,
        # and this line ensures that this component is also made deterministic
        # Gymnasium does NOT support env.seed()
        # Seeding is done during reset
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except Exception:
        pass

 # calculation of epsilon as it decrease frame by frame from (1 to 0.05)
 #  More frames → agent has learned more → explore less. update after transition to next state
def epsilon_per_fram(frame_idx):
    epsilon = end_eps + (start_eps - end_eps) * np.exp(-1.0 * frame_idx / eps_decay)
    return float(epsilon)


####   Training FUnction   ####

def training():

    env = gymnasium.make("CartPole-v1")
    set_seed(env, seed)

    # for our environment this is sufficient
    # state_dim = env.observation_space.shape[0]
    # n_actions = env.action_space.n

    # for complex (3d) env
    obs_space = env.observation_space
    # Ensures the environment returns a continuous vector state
    assert isinstance(obs_space, gymnasium.spaces.Box), "This implementation assumes Box state space (vector)."
    state_shape = obs_space.shape
    state_dim = state_shape[0]
    n_actions = env.action_space.n

    # model
    online_net = QNetwork(state_dim,n_actions,dueling=True).to(device)
    target_net = QNetwork(state_dim,n_actions,dueling=True).to(device)
    #PyTorch “state dictionary” ->Contains all network parameters (weights + biases).
    # online_net.state_dict() -> extracts all learned weights.
    target_net.load_state_dict(online_net.state_dict())
    # target_net.load_state_dict(...)→ loads them into target network.
    target_net.eval() # freeze network

    optimizer = optim.Adam(online_net.parameters(),lr=alpha)
    loss_function = nn.SmoothL1Loss()

    buffer = ReplayBuffer(buffer_capacity,state_dim)

    frame_idx = 0
    all_reward =[]
    losses = []
    episode = 0 # using while loop

    state = env.reset()  # reset the environment for every episode as it should start from starting
    if isinstance(state, tuple):
        state, _ = state  # new version return state , info(return starting state)

    for _ in range(min_replay_size):
        action = env.action_space.sample()
        step = env.step(action)
        if len(step) == 5 :
            next_state , reward , terminate ,trunc, _ = step
            done = trunc or terminate
        else :
            next_state , reward , done , _ = step
        buffer.add(state ,next_state , action , reward , done)
        state = next_state if not done else env.reset() # env reset will again return new state there for
        if isinstance(state,tuple):
            state , _ = state

    print(f"Replay buffer initialized ({len(buffer)} transitions). Starting training...")

    while episode < episodes:
        reset_output = env.reset()
        if isinstance(reset_output,tuple):
            state , _ = reset_output
        else :
            state = reset_output

        episode_reward = 0.0
        for step in range(max_step_per_episode ):
            frame_idx += 1
            epsilon = epsilon_per_fram(frame_idx)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = online_net(state_tensor)
                    action = int(q_vals.argmax(dim=1).item())
            step_output = env.step(action)
            if len(step_output) == 5 :
                next_state , reward , terminate ,trunc, _ = step_output
                done = trunc or terminate
            else :
                next_state , reward , done , _ = step_output
            buffer.add(state,next_state,action ,reward,done)
            state = next_state
            episode_reward += reward
            if done:
                break

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states = torch.FloatTensor(batch['state']).to(device) # [B, state_dim]
                actions = torch.LongTensor(batch['action']).unsqueeze(1).to(device) # [B, 1]
                rewards = torch.FloatTensor(batch['reward']).unsqueeze(1).to(device) # [B, 1]
                next_states = torch.FloatTensor(batch['next_state']).to(device) # [B, state_dim]
                dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(device)# [B, 1]

                q_value = online_net(states)
                q_sa = q_value.gather(1,actions)# gather(1, actions) picks the Q-value of ONLY the action actually taken.
                # q_value return numbers of action but using gather we only store what we are selecting

                # we are using online_net for selecting action and target_net for evaluating that action
                with torch.no_grad():
                    if use_double:
                        next_q_value_online = online_net(next_states) # Online network decides which action is best as it reflect current policy
                        best_action = next_q_value_online.argmax(dim = 1,keepdim=True)
                        next_q_value_target = target_net(next_states) # Target network evaluates the value provided by online network
                        max_next_q_value= next_q_value_target.gather(1,best_action)
                    else:
                        next_q_value_target = target_net(next_states)
                        max_next_q_value , _ = next_q_value_target.max(dim = 1,keepdim=True)

                    target = rewards + (1 - dones) * gamma * max_next_q_value

                loss = loss_function(q_sa ,target)

                optimizer.zero_grad()
                loss.backward()
                # we use gradient clipping as one bad batch can explode
                # bootstrapped  value == target
                # Gradient clipping is required in DQN to stabilize bootstrapped value learning by preventing exploding gradients caused by overestimated Q-targets
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)# gradiant clipping
                optimizer.step()
                losses.append(loss.item())

                #This happens once at the beginning.It makes the two networks identical so training can start cleanly.
                # But after training begins, the online network changes every gradient step, and the target network must lag behind.
                # if frame_idx % target_update_freq == 0 : # hard update
                #     target_net.load_state_dict(online_net.state_dict())
                # change using gpt as cart pole is small environment it does not need hard update
                # there for we use this soft update
                tau = 0.005
                for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                    target_param.data.copy_(
                        tau * online_param.data + (1 - tau) * target_param.data
                    )
                if done:
                    break

        all_reward.append(episode_reward)
        episode += 1

        if episode % 10 == 0:
            avg_reward = np.mean(all_reward[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Ep {episode:4d} | AvgRew(last10) {avg_reward:6.2f} | Frames {frame_idx} | "
                  f"Eps {epsilon:.3f} | AvgLoss {avg_loss:.4f}")

    env.close()
    return online_net, target_net, all_reward, losses

if __name__ == "__main__":
    net, target, rewards, losses = training()
    print("Training finished.")