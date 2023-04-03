from network import deepQnet
from replay_buffer import ReplayMemory
import numpy as np
import random
import math
import torch
from torch import nn
from collections import namedtuple


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
                action=policy_net(state).max(1)[1].view(1, 1)
            return action
    
    def greedy(self,state,policy_net):
        with torch.no_grad():
            action=policy_net(state).max(1)[1].view(1, 1)
        return action
    
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
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LR,amsgrad=True)

    def init_policy(self,eps_start,eps_end,eps_decay,env):
        self.policy=Policy(eps_start,eps_end,eps_decay,env)

    def envStep(self,state):
        state_tensor=torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        action=self.policy.epsilon_greedy(state_tensor,self.policy_net)
        self.policy.epsilon_update() 
        next_state, reward, done = self.env.step(action.item())
        if done:
            next_state=None
        transition=Transition(state,action,reward,next_state)
        self.memory.push(transition)
        return next_state, done

    def optimizationStep(self,transition_batch):
        next_states_batch=transition_batch.next_state
        targets_batch=torch.zeros(self.batch_size)
        targets_batch+=torch.tensor(transition_batch.reward)
        terminal_state_mask=list(map(lambda s: s is not None, next_states_batch))
        next_states_batch=torch.tensor(np.array([ns for ns in transition_batch.next_state if ns is not None]),dtype=torch.float32)
        with torch.no_grad():
            targets_batch[terminal_state_mask]+=self.gamma*torch.max(self.target_net(next_states_batch),1)[0]
        actions_batch=torch.tensor(transition_batch.action)
        states_batch=torch.tensor(np.array([s for s in transition_batch.state]),dtype=torch.float32)
        state_action_batch=torch.gather(self.policy_net(states_batch),1,actions_batch.unsqueeze(1)).squeeze(1)

        criterion=nn.SmoothL1Loss()
        loss=criterion(state_action_batch,targets_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self,n_episodes):
        global_step_counter=0
        for episode in range(n_episodes):
            observation = self.env.reset()
            while True:
                next_observation, done=self.envStep(observation)
                if len(self.memory)<self.batch_size:
                    if done:
                        break
                    continue
                else:
                    transitions=self.memory.sample(self.batch_size)
                    transition_batch = Transition(*zip(*transitions))
                self.optimizationStep(transition_batch)

                if done:
                    break

                observation=next_observation
                global_step_counter+=1
                if global_step_counter%self.target_reset==0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
    