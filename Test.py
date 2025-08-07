import TRL 
import torch as th
import torch.nn as nn
import random
import numpy as np
seed = 1
th.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env = TRL.UnityEnvAdapterVisCont("ML-env-No-Red/ML-env.exe",seed, 100)
#env = TRL.UnityEnvAdapterVisCont(None)
names = set(env.behavior_name)

class IPPOActor(nn.Module):
    def __init__(self, env: TRL.Enviroment_Base):
        super().__init__()
        
        sh = env.get_action_space()[env.behavior_name[0]]["shape"]
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # → 32×31×31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → 64×14×14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → 64×12×12
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # → 128×10×10
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),# → 128×8×8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),                 # → 8192 → 512
            nn.ReLU()
        )

        self.mu_head = nn.Linear(512, sh)
        self.sigma_head = nn.Linear(512, sh)

    def forward(self, x):
        x = self.feature_extractor(x)
        mu = self.mu_head(x)
        sigma = th.exp(self.sigma_head(x)) 
        return mu, sigma
    
class IPPOCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # → 32×31×31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → 64×14×14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → 64×12×12
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # → 128×10×10
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),# → 128×8×8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),                 # → 8192 → 512
            nn.ReLU()
        )
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        value = self.value_head(x)
        return value
    
learner = TRL.Learner_IPPO(names, env.get_observation_space(), env.get_action_space(), IPPOActor(env), IPPOCritic())
logger = TRL.Logger_Base()

print("setting up trainer")
trainer = TRL.Trainer_Basic(env, learner, logger)
print("starting training")
trainer.start()