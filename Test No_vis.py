import TRL 
import torch as th
import torch.nn as nn
import random
import numpy as np
seed = 1
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env = TRL.UnityEnvAdapterVecCont("ML-env-No-Vis-No-Red/ML-env.exe",seed, 100, graphics=False)
#env = TRL.UnityEnvAdapterVecCont(None, seed, 100)
names = set(env.behavior_name)

class IPPOActor(nn.Module):
    def __init__(self, env: TRL.Enviroment_Base):
        super().__init__()
        
        sh = env.get_action_space()[env.behavior_name[0]]["shape"]
        self.feature_extractor = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU(),
            nn.Linear(256, 256),                 
            nn.ReLU()
        )

        self.mu_head = nn.Linear(256, sh)
        self.sigma_head = nn.Linear(256, sh)

    def forward(self, x):
        x = self.feature_extractor(x)
        mu = self.mu_head(x)
        sigma = th.nn.functional.softplus(self.sigma_head(x)) + 1e-3
        return mu, sigma
    
class IPPOCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(2,256),
            nn.ReLU(),
            nn.Linear(256, 256),                 
            nn.ReLU()
        )
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        value = self.value_head(x)
        return value
    
learner = TRL.Learner_IPPO(names, env.get_observation_space(), env.get_action_space(), IPPOActor(env), IPPOCritic())
logger = TRL.Logger_Base()

print("setting up trainer")
trainer = TRL.Trainer_OnPolicy(env, learner, logger, runcount=10)
print("starting training")
trainer.start()