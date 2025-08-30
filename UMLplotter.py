import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import pandas as pd
import matplotlib.cm as cm


In_Path = Path(__file__, '..','UML_input')
save_path = Path(__file__, '..','UML_output')
file = os.listdir(In_Path)[0]

df = pd.read_csv(Path(In_Path, file))
print(df)
steps = df['Step'].astype(float) 
mean = df['Value'].astype(float)
plt.plot(steps, mean, label='Mean Reward', color='blue')
plt.xlabel('Environment Steps')
plt.ylabel('Reward')
plt.ylim(bottom=0)
plt.title('Shared Reward')
plt.legend()
plt.savefig(Path(save_path, 'reward_plot.png'))
plt.clf()