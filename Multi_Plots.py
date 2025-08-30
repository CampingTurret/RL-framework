import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import pandas as pd
import matplotlib.cm as cm


fig_reward, ax_reward = plt.subplots(figsize=(10, 6))
fig_entropy_blue, ax_entropy_blue = plt.subplots(figsize=(10, 6))
fig_entropy_red, ax_entropy_red = plt.subplots(figsize=(10, 6))

search_config = {               
    "offpolicy_iterations": None,              
    "entropy_loss_param": 5.0e-3   ,      #5.0e-3     
    "lr": 1e-4 ,                #1e-5    
    "value_param": 1              
}
pre_ls = []
Single_selection = False
if search_config["entropy_loss_param"] is not None:
    pre_ls.append(f"{search_config['entropy_loss_param']}")
    if search_config["lr"] is not None:
        pre_ls.append(f"{search_config['lr']}")
        if search_config["offpolicy_iterations"] is not None:
            pre_ls.append(f"{search_config['offpolicy_iterations']}")
            if search_config["value_param"] is not None:
                pre_ls.append(f"{search_config['value_param']}")
                Single_selection = True
                post_ls = []
            else:
                post_ls = []
        else:
            post_ls = [f'{search_config["value_param"]}']
    else:
        post_ls = [f'{search_config["offpolicy_iterations"]}', f'{search_config["value_param"]}']
else:
    post_ls = [f'{search_config["lr"]}', f'{search_config["offpolicy_iterations"]}', f'{search_config["value_param"]}']
Pre_path = Path(__file__, '..','Data', *pre_ls)



folder_names = [f for f in os.listdir(Pre_path)]
colors = cm.get_cmap('tab10', len(folder_names))

if not Single_selection:
    for idx, folder in enumerate(folder_names):
        load_path = Path(Pre_path, folder, *post_ls)
        print(load_path)
        reward_agent = 'BlueAgent'
        rewards = []
        entropy_blue = []
        entropy_red = []
        try:
            for f in os.listdir(load_path):
                a = f
                name_id = a.replace('.csv', '').replace('-team-0', '').split('_')
                df = pd.read_csv(Path(load_path, f))
                if name_id[0] == reward_agent:
                    print(name_id[1], ':', len(df))
                    rewards.append(df[['env_step','reward_mean']])
                    entropy_blue.append(df[['env_step','mean_entropy']])
                elif name_id[0] == 'RedAgent':
                    entropy_red.append(df[['env_step','mean_entropy']])
        except Exception as e:
            if isinstance(e, OSError):
                print('runs do not exist')
                continue
            else: raise e
        label = folder
        color = colors(idx)
        if len(rewards) != 0:
            combined = pd.concat(rewards, ignore_index=True)
            summary = combined.groupby('env_step')['reward_mean'].agg(['mean', 'std']).reset_index()
            summary.columns = ['env_step', 'mean_reward', 'std_reward']
            summary.reset_index(inplace=True)
            rewards = summary
            ax_reward.plot(summary['env_step'], summary['mean_reward'], label=label, color=color)
            ax_reward.fill_between(summary['env_step'], summary['mean_reward'] - summary['std_reward'], summary['mean_reward'] + summary['std_reward'], color=color, alpha=0.2)
        if len(entropy_red) != 0:
            combined = pd.concat(entropy_red, ignore_index=True)
            summary = combined.groupby('env_step')['mean_entropy'].agg(['mean', 'std']).reset_index()
            summary.columns = ['env_step', 'mean_entropy', 'std_entropy']
            summary.reset_index(inplace=True)
            entropy_red = summary
            ax_entropy_blue.plot(summary['env_step'], summary['mean_entropy'], label=label, color=color)
            ax_entropy_blue.fill_between(summary['env_step'], summary['mean_entropy'] - summary['std_entropy'], summary['mean_entropy'] + summary['std_entropy'], color=color, alpha=0.2)

        if len(entropy_blue) != 0:
            combined = pd.concat(entropy_blue, ignore_index=True)
            summary = combined.groupby('env_step')['mean_entropy'].agg(['mean', 'std']).reset_index()
            summary.columns = ['env_step', 'mean_entropy', 'std_entropy']
            summary.reset_index(inplace=True)
            entropy_blue = summary
            ax_entropy_red.plot(summary['env_step'], summary['mean_entropy'], label=label, color=color)
            ax_entropy_red.fill_between(summary['env_step'], summary['mean_entropy'] - summary['std_entropy'], summary['mean_entropy'] + summary['std_entropy'], color=color, alpha=0.2)

save_path = Path(__file__).parent / 'Multi_Plots'
save_path.mkdir(exist_ok=True)

# Reward
ax_reward.set_title('Reward Comparison Across Configurations')
ax_reward.set_xlabel('Environment Steps')
ax_reward.set_ylabel('Reward')
ax_reward.legend()
ax_reward.set_ylim(bottom=0)
fig_reward.savefig(save_path / 'reward_comparison.png')
plt.close(fig_reward)

# Entropy Blue
ax_entropy_blue.set_title('Blue Agent Entropy Comparison')
ax_entropy_blue.set_xlabel('Environment Steps')
ax_entropy_blue.set_ylabel('Entropy')
ax_entropy_blue.legend()
ax_entropy_blue.set_ylim(top=5)
fig_entropy_blue.savefig(save_path / 'entropy_blue_comparison.png')
plt.close(fig_entropy_blue)

# Entropy Red
ax_entropy_red.set_title('Red Agent Entropy Comparison')
ax_entropy_red.set_xlabel('Environment Steps')
ax_entropy_red.set_ylabel('Entropy')
ax_entropy_red.set_ylim(top=5)
ax_entropy_red.legend()
fig_entropy_red.savefig(save_path / 'entropy_red_comparison.png')
plt.close(fig_entropy_red)