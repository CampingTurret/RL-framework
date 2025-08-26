import TRL
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import pandas as pd

class IPPOActor(nn.Module):
    def __init__(self, env: TRL.Enviroment_Base):
        super().__init__()
        
        sh = env.get_action_space()[env.behavior_name[0]]["shape"]
        sb = env.get_observation_space()[env.behavior_name[0]]["shape"]

        w = ((((sb[1] - 8)//4 + 1) - 4)//2 + 1) - 3 + 1
        h = ((((sb[2] - 8)//4 + 1) - 4)//2 + 1) - 3 + 1

        # Mirroring Unity ML Simple structure
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  
            nn.ReLU(),
            nn.Flatten()
        )
        self.Policy_net = nn.Sequential(
            nn.Linear(64*w*h, 512),                 
            nn.ReLU(),
            nn.Linear(512, 512),                 
            nn.ReLU()
        )
        self.Value_net = nn.Sequential(
            nn.Linear(64*w*h, 128),                 
            nn.ReLU(),
            nn.Linear(128, 128),                 
            nn.ReLU()
        )
        

        # Mu and Sigma heads and then the Value head for shared backbone
        self.mu_head = nn.Linear(512, sh)
        self.sigma_head = nn.Linear(512, sh)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.Policy_net(x)
        z = self.Value_net(x)
        mu = self.mu_head(y)
        # check how Unity ML does it: 
        mu = th.tanh(mu)
        sigma = th.exp(self.sigma_head(y)) 
        sigma = th.maximum(sigma, th.tensor(1e-1))
        value = self.value_head(z)
        return mu, sigma, value
    

pyrun_path = Path(__file__, '..','Pyruns')
for id in os.listdir(pyrun_path):
    fname_base = Path(__file__, '..','Pyruns', f'{id}')
    dirls = os.listdir(fname_base)
    x = 'final'
    if 'final' not in dirls:
        dirls.sort(key= lambda x: int(x), reverse=True)
        x = dirls[0]
    if x == 'final':
        saved_dict = th.load(Path(__file__, '..','Pyruns', f'{id}', f'{x}', 'checkpoint_final.pkl' ), weights_only=False)
    else: 
        saved_dict = th.load(Path(__file__, '..','Pyruns', f'{id}', f'{x}', 'checkpoint.pkl' ), weights_only=False)
    logger_data = saved_dict['logger'].dataframes


    ## Reward plots:
    agent1 = next(iter(saved_dict['logger'].dataframes))
    steps = logger_data[agent1]['env_step'].astype(float) 
    mean = logger_data[agent1]['reward_mean'].astype(float)
    std = logger_data[agent1]['reward_std'].astype(float)
    plt.plot(steps, mean, label='Mean Reward', color='blue')
    plt.fill_between(steps, mean - std, mean + std, color='blue', alpha=0.2, label='±1 Standard Deviation')
    plt.xlabel('Environment Steps')
    plt.ylabel('Reward')
    plt.ylim(bottom=0)
    plt.title('Shared Reward Mean ± Std Dev')
    plt.legend()
    plt.show()
    print(f'{id}:{x}:{logger_data}')