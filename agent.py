from network import deepQnet
from replay_buffer import ReplayMemory
import numpy as np
import random
import math
import torch
from torch import nn
from collections import namedtuple
from tqdm import tqdm


Transition=namedtuple('Transition','state action reward next_state')

class Policy:

    def __init__(self,eps_start,eps_end,eps_decay,env):
        """
        The policy class.
        param eps_start: initial value of the decaying epsilon parameter of epsilon-greedy policy
        param eps_end: asymptotic lower bound value of the decaying epsilon parameter of epsilon-greedy policy 
        param eps_decay: controls decay rate for epsilon parameter of epsilon-greedy policy
        param env: environment
        """
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps=eps_start
        self.eps_decay=eps_decay
        self.steps=0
        self.env=env

    def epsilon_greedy(self,state,policy_net):
        if random.random() < self.eps:
            return torch.tensor([[random.randrange(self.env.num_actions)]], dtype=torch.long)
        else:
            with torch.no_grad():
                # action=policy_net(state).max(1)[1].view(1, 1)
                actions=(policy_net(state).squeeze(1)).squeeze(1)
                action=actions.max(1)[1].view(1, 1)
            return action
    
    def greedy(self,state,policy_net):
        with torch.no_grad():
            # action=policy_net(state).max(1)[1].view(1, 1)
            actions=(policy_net(state).squeeze(1)).squeeze(1)
            action=actions.max(1)[1].view(1, 1)
        return action.item()
    
    def epsilon_update(self):
        # call after every step to decay epsilon
        self.steps+=1
        self.eps=self.eps_end+(self.eps_end-self.eps_start)*math.exp(-1*self.steps/self.eps_decay)

class Agent:

    def __init__(self,env,batch_size,buffer_length,LR,gamma,target_reset):
        """
        The agent class.
        param env: environment
        param batch_size: batch size used for training
        param buffer_length: maximum length of replay buffer
        param LR: learning rate for the state-action value update
        param gamma: discount factor of the pendulum MDP
        param target_reset: number of steps after which target Q-network is updated
        """
        self.env=env
        self.LR=LR
        self.batch_size=batch_size
        self.memory=ReplayMemory(buffer_length)
        self.gamma=gamma
        self.target_reset=target_reset

    def build_net(self):
        self.policy_net=deepQnet(len(self.env.s),self.env.num_actions)
        self.target_net=deepQnet(len(self.env.s),self.env.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.LR)

    def init_policy(self,eps_start,eps_end,eps_decay,env):
        self.policy=Policy(eps_start,eps_end,eps_decay,env)

    def envStep(self,state):
        action=self.policy.epsilon_greedy(state,self.policy_net)
        self.policy.epsilon_update() 
        next_state, reward, done = self.env.step(action.item())
        reward = torch.tensor([reward])
        if done:
            next_state=None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        transition=Transition(state,action,reward,next_state)
        self.memory.push(transition)
        return next_state, reward.item(), done

    def optimizationStep(self,transition_batch):
        next_states_batch=transition_batch.next_state
        terminal_state_mask=torch.tensor(tuple(map(lambda s: s is not None, next_states_batch)),dtype=torch.bool)
        non_terminal_next_states_batch=torch.cat([ns for ns in transition_batch.next_state if ns is not None])
        next_state_values=torch.zeros(self.batch_size) # values are discounted
        with torch.no_grad():
            next_state_values[terminal_state_mask]+=self.gamma*torch.max(self.target_net(non_terminal_next_states_batch),1)[0]
        actions_batch=torch.cat(transition_batch.action)
        states_batch = torch.cat(transition_batch.state)
        rewards_batch = torch.cat(transition_batch.reward)
        targets_batch=next_state_values+rewards_batch
        state_action_batch=torch.gather(self.policy_net(states_batch),1,actions_batch).squeeze(1)

        criterion=nn.MSELoss()
        loss=criterion(state_action_batch,targets_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self,n_episodes):

        data = {
        'return': [],
        'reward' : [],
        'avg_eps_reward' : [],
        'real_time':[0.1]}

        global_step_counter=0
        for episode in tqdm(range(n_episodes),desc="Training Progress"):
            observation = self.env.reset()
            observation = torch.tensor(observation,dtype=torch.float32).unsqueeze(0) #newww
            G=0
            r_sum=0
            eps_step=0
            while True:
                next_observation,reward, done=self.envStep(observation)   # changed reward added output
                G+=reward*self.gamma**(eps_step)
                eps_step+=1
                r_sum+=reward
                data['reward'].append(reward)
                if len(self.memory)<self.batch_size:
                    if done:
                        data['return'].append(G)
                        data['avg_eps_reward'].append(r_sum/eps_step)
                        break
                    continue
                else:
                    transitions=self.memory.sample(self.batch_size)
                    transition_batch = Transition(*zip(*transitions))
                self.optimizationStep(transition_batch)
                if done:
                    data['return'].append(G)
                    data['avg_eps_reward'].append(r_sum/eps_step)
                    break
                data['real_time'].append(data['real_time'][-1]+self.env.dt)
                observation=next_observation
                global_step_counter+=1
                if global_step_counter%self.target_reset==0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        return data
    