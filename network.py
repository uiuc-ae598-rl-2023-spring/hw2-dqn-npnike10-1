from torch import nn
import torch.nn.functional as F
import torch

class deepQnet(nn.Module):
            def __init__(self,n_observations,n_actions):
                super().__init__()
                self.layer1=nn.Linear(n_observations,64)
                self.layer2=nn.Linear(64,64)
                self.layer3=nn.Linear(64,n_actions)
                
            def forward(self,x):
                x=torch.tanh(self.layer1(x))
                x=torch.tanh(self.layer2(x))
                return self.layer3(x)