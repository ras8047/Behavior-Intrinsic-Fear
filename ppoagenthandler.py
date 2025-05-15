import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import random 
import os 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
        
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()
        
    def parse_states(self,batch_size):
        states=[]
        next_states=[]
        for i in range(len(self.states)-1):
            states.append(self.states[i])
            next_states.append(self.states[i])
        local_states=random.sample(states, batch_size)
        local_next_state=random.sample(next_states, batch_size)
        return states,next_states,local_states,local_next_state


class Critic(nn.Module):
    def __init__(self, input_shape,action_dim):
        super().__init__()
        obs_shape = input_shape
        self.conv = nn.Sequential(layer_init(nn.Conv2d(obs_shape[2], 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(3136, 512)))

        self.fc1 = layer_init(nn.Linear(512, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc_q = layer_init(nn.Linear(64, 1))

    def forward(self, x):
        x = self.conv(x / 255.0)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        q_vals = self.fc_q(x)
        return q_vals
    
    
    def feat_forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x


class ActorCritic(nn.Module):
    def __init__(self, envs):
        super(ActorCritic, self).__init__()
        obs_shape = envs.observation_space.shape
        action_dim=envs.action_space.n
        #actor
        self.actor = nn.Sequential(
                        layer_init(nn.Conv2d(obs_shape[2], 32, 8, stride=4)),
                        nn.ReLU(),
                        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                        nn.ReLU(),
                        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                        nn.ReLU(),
                        nn.Flatten(),
                        layer_init(nn.Linear(3136,512)),
                        nn.ReLU(),
                        nn.Linear(512, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )

        
        self.critic = Critic(obs_shape,action_dim)

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):


        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, envs,args):
        self.args=args
        self.envs=envs
        self._set_memory()
        self._set_networks()
        self._set_seed()
        # self._set_rnd()
    
    def _set_memory(self):
        self.buffer = RolloutBuffer()

    
    def _set_networks(self):
        self.policy = ActorCritic(self.envs).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.args.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.args.lr_critic}])

        self.policy_old = ActorCritic(self.envs).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    
    def _set_seed(self):
        fixed_digits = 6 
        self.seed=(random.randrange(111111, 999999, fixed_digits))
        torch.manual_seed(self.seed)
        # env.seed(self.seed)
        np.random.seed(self.seed)

    def act(self, state):
        with torch.no_grad():
            if len(np.shape(state))<=3:
                state=np.reshape(state,(1,*(state.shape)))
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()
    
    def remember(self,reward,done):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)


    def train(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()
        

        for _ in range(self.args.k_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()

        return loss,ratios,0
    
    def save_state(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.policy_old.state_dict(), file_path+"//old_policy")
        torch.save(self.policy.state_dict(), file_path+"//policy")

        torch.save(self.optimizer.state_dict(), file_path +"//optimizer")

    def load_state(self,file_path):
        self.policy_old.load_state_dict(torch.load(file_path+"//old_policy" ))
        self.policy.load_state_dict(torch.load(file_path+"//policy"))

        self.optimizer.load_state_dict(torch.load(file_path +"//optimizer"))







        
        
        
