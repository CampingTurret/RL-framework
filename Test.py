import TRL 
import torch as th
import torch.nn as nn
import random
import numpy as np
seed = random.randint(0,999)
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(seed)
env = TRL.UnityEnvAdapterVisCont("ML-env/ML-env.exe",seed, 500)
#env = TRL.UnityEnvAdapterVisCont(None, seed, 100)
names = set(env.behavior_name)

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
        mu = mu
        sigma = th.exp(self.sigma_head(y))
        value = self.value_head(z)
        return mu, sigma, value
    

config = {
    "id" : 80,
    "gamma": 0.99,                
    "offpolicy_iterations": 2,     
    "grad_norm_clip": 1,           
    "entropy_loss_param": 5.0e-3,  
    "ppo_clip_eps": 0.2,          
    "lr": 1e-4,                    
    "value_param": 1               
}
learner = TRL.Learner_IPPO(names, env.get_observation_space(), env.get_action_space(), IPPOActor, tuple([env]), config=config)
logger = TRL.Logger_PPO()

print("setting up trainer")
trainer = TRL.Trainer_OnPolicy(env, learner, logger, runcount=5)
print("starting training")
trainer.start()