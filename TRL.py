import torch as th
from abc import ABC, abstractmethod
from collections.abc import Mapping
import mlagents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import gym
import copy
from transition_batch import TransitionBatch
import matplotlib.pyplot as plt

class Enviroment_Base(ABC):

    _num_agents: int
    behavior_name: list[str]
    max_len: int

    @abstractmethod
    def reset(self) -> dict[str,np.ndarray]:
        """Resets the environment and returns the initial observation."""
        pass

    @abstractmethod
    def step(self, actions: dict[str,np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]]:
        """
        Takes a step in the environment.
        Returns: obs, rewards, dones, infos
        """
        pass

    @abstractmethod
    def get_observation_space(self)->dict[str, dict]:
        pass

    @abstractmethod
    def get_action_space(self)->dict[str, dict]:
        pass

    @property
    def num_agents(self):
        return self._num_agents
    

class Learner_Base(ABC):

    names: set
    obs_spec: dict
    action_spec: dict
    gamma: float

    def __init__(self, names, obs_spec, action_spec) -> None:
        super().__init__()
        self.names = names
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    @abstractmethod
    def eval(self, obs) -> dict[str,np.ndarray]:
        pass

    @abstractmethod
    def learn(self, batch, name = 'agent') -> dict[str, float]:
        pass

class Logger_Base(ABC):
    def __init__(self) -> None:
        super().__init__()


    def log_agent_names(self, names:set):
        for name in names:
            print(name)

    def log_train_results(self, dict_:dict):
        for k, v in dict_.items():
            tmp = [f'{l}:{format(p, ".4g")}' for (l,p) in v.items()]
            print(f'{k} -> ' + '-'.join(tmp))



class Trainer_Base(ABC):
    def __init__(self, env: Enviroment_Base, learner: Learner_Base, logger: Logger_Base):
        self.env = env
        self.learner = learner
        self.logger = logger
        self.obs_space = env.get_observation_space()
        self.act_space = env.get_action_space()
        p = set()
        for name in self.obs_space.keys():
            p.add(name)
        self.agents = p
        self.logging: str|None

    def start(self, logging: str | None = 'verbose'):
        if logging == 'verbose':
            self.logger.log_agent_names(set(self.obs_space.keys()))
        self.obs = self.env.reset()
        while True:
            self.train()
    

    @abstractmethod
    def train(self):
        pass


    

class Trainer_OnPolicy(Trainer_Base):

    def __init__(self, env: Enviroment_Base, learner: Learner_Base, logger: Logger_Base):
        super().__init__(env, learner, logger)

    def batch_form(self, name):
        return {'actions': (self.act_space[name]["shape"], th.float32),
                'states': (self.obs_space[name]["shape"], th.float32),
                'next_states': (self.obs_space[name]["shape"], th.float32),
                'rewards': ((1,),  th.float32),
                'dones': ((1,), th.bool),
                'returns': ((1,), th.float32)}

    def train(self):
        transes = {x:TransitionBatch(self.env.max_len,self.batch_form(x)) for x in self.agents}
        infodict = {}
        for act in self.agents:
            infodict[act] = {}
        done_main = False
        self.obs = self.env.reset()
        for x in range(self.env.max_len):
            actions = self.learner.eval(self.obs)
            step = self.env.step(actions)
            for act in step.keys():
                obs, rewards, dones, infos = step[act]
                img = (obs[0,:,:,:]*255).astype(np.int32)
                print(img[:,0,0])
                print(img[:,1,0])
                print("Min:", img.min(), "Max:", img.max(), "Mean:", img.mean())
                plt.imshow(img)
                plt.show()
                dic = {'actions': th.tensor(actions[act], dtype=th.float32),
                                  'states': th.tensor(self.obs[act], dtype=th.float32),
                                  'next_states': th.tensor(obs, dtype=th.float32),
                                  'rewards': th.tensor(rewards, dtype=th.float32).reshape((1,1)),
                                  'dones': th.tensor(dones, dtype=th.bool).reshape((1,1))}
                transes[act].add(dic)
                if np.any(dones) or x == (self.env.max_len - 1):
                    done_main = True
                    infodict[act]["episode_length"] = transes[act].size
                    infodict[act]["episode_reward"] = th.sum(transes[act]['rewards'])
                

            if done_main or x == (self.env.max_len - 1):
                for act in self.agents:
                    transes[act]['returns'][x] = transes[act]['rewards'][x]
                    for i in range(x - 1, - 1, -1):
                        transes[act]['returns'][i] = transes[act]['rewards'][i] + self.learner.gamma * transes[act]['returns'][i + 1]
                break
        print("Ran")
        for act in self.agents:
            print(f"learning:{act}")
            infodict[act].update(self.learner.learn(transes[act], act))
        self.logger.log_train_results(infodict)        

#Needs Work
class Learner_IPPO(Learner_Base):

    actor: dict[str, th.nn.Module]
    optims: dict[str, th.optim.Optimizer]
    batch: dict[str, TransitionBatch]

    def __init__(self, names, obs_spec, action_spec, actor_network:th.nn.Module, critic_network:th.nn.Module) -> None:
        super().__init__(names, obs_spec, action_spec)
        p = {}
        self.gamma = 0.99
        self.offpolicy_iterations = 0
        self.grad_norm_clip = 1
        self.entropy_loss_param = 1
        self.ppo_clip_eps = 0.2
        self.lr = 1e-5
        self.lrc = 1e-5
        self.actor = {
            agent: copy.deepcopy(actor_network) for agent in self.names
        }
        self.critic = {
            agent: copy.deepcopy(critic_network) for agent in self.names
        }
        self.optims = {
            agent:th.optim.Adam(self.actor[agent].parameters(), lr=self.lr) for agent in self.names
        }
        self.optims_c = {
            agent:th.optim.Adam(self.critic[agent].parameters(), lr=self.lrc) for agent in self.names
        }
    def _advantages(self, batch, values=None, next_values=None, lambda_=0.95):
        rewards = batch['rewards']
        masks = batch['dones'] # 1 if done, else 0
        T = rewards.size(0)

        deltas = rewards + self.gamma * next_values * ~masks - values
        advantages = th.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * lambda_ * ~masks[t] * gae
            advantages[t] = gae

        return advantages

    #def _advantages(self, batch, values=None, next_values=None):
    #    return batch['rewards'] + self.gamma * next_values - values

    def _policy_loss(self, ratio, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.old_pi is None:
            return -(advantages.detach()).mean()
        return -th.minimum((advantages.detach() * ratio), advantages.detach() * th.clip(ratio, 1 - self.ppo_clip_eps, 1+self.ppo_clip_eps)).mean()
        
    def _value_loss(self, batch, values, next_values):
        return ((batch['rewards'][:-1] + self.gamma * next_values[:-1] - values[:-1])**2).mean()

    def _entropy_loss(self, pi):
        return pi.entropy().mean()

        
    def eval(self, obs) -> dict[str,np.ndarray]:
        d = {}
        for name in obs.keys():
            out = self.actor[name](th.tensor(obs[name]).detach())
            v = th.distributions.Normal(out[0],out[1])
            d.update({name:(th.tanh(v.sample())).detach().numpy()})
        return d

    def probabity(self, obs, name):
        model = self.actor[name]
        out = model(obs)
        v = th.distributions.Normal(out[0],out[1])
        return v

    def learn(self, batch, name = 'agent'):
        model = self.actor[name]
        model.train(True)
        self.old_pi = None
        loss_sum = 0.0
        avg_entropy = 0
        avg_policy = 0
        avg_value = 0
        for _ in range(1 + self.offpolicy_iterations):
            
            val = self.critic[name](batch['states'])
            next_val = self.critic[name](batch['next_states'])

            pi = self.probabity(batch['states'], name)
            ratio = th.exp(pi.log_prob(batch['actions']).detach() - pi.log_prob(batch['actions']) ) if self.old_pi is None else th.exp((pi.log_prob(batch['actions']) - self.old_pi.log_prob(batch['actions'])))
            policy = self._policy_loss(ratio, self._advantages(batch, val, next_val))
            entropy = self._entropy_loss(pi)
            loss = policy + self.entropy_loss_param * entropy
            loss_c = self._value_loss(batch, val, next_val) 
            if self.old_pi is None:
                self.old_pi = th.distributions.Normal(pi.mean.detach(), pi.stddev.detach())
            # Backpropagate loss
            self.optims[name].zero_grad()
            self.optims_c[name].zero_grad()
            loss.backward()
            loss_c.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
            self.optims[name].step()
            self.optims_c[name].step()
            avg_entropy += entropy.item()
            avg_policy += policy.item()
            avg_value += loss_c.item()
            loss_sum += loss.item() + loss_c.item()
        return {"loss":loss_sum/(self.offpolicy_iterations+1), "entropy":avg_entropy/(self.offpolicy_iterations+1), "policy":avg_policy/(self.offpolicy_iterations+1), "value":avg_value/(self.offpolicy_iterations+1)}


class GymEnvAdapterCont(Enviroment_Base):
    def __init__(self, env_id, max_len = 1000):
        self.max_len = 1000
        self.env = gym.make(env_id)
        self._num_agents = 1
        self.behavior_name = ["Agent"]

    def reset(self):
        obs, info = self.env.reset()
        obs = np.expand_dims(obs, axis=0)  
        self.obs = obs
        return {self.behavior_name[0]:obs}

    def step(self, actions):
        obs, reward, done, trunc, info = self.env.step(actions)
        obs = np.expand_dims(obs, axis=0)  
        reward = np.array([reward])
        done = np.array([done or trunc]) 
        info = [info] 
        ret =  (obs, reward, done, info)
        return {self.behavior_name[0]:ret}


    def get_observation_space(self):
        return {"Agent":{"shape" :self.env.observation_space.shape, "type":"any"}}

    def get_action_space(self):
        return {"Agent":{"type": "continuous", "size": self.env.action_space.shape}}

class UnityEnvAdapterVisCont(Enviroment_Base):
    def __init__(self, env_path, seed=1, max_len = 1000):
        self.max_len = max_len
        self.env = UnityEnvironment(file_name=env_path, seed=seed, no_graphics=True)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())
        self.spec = {name:self.env.behavior_specs[name] for name in self.behavior_name}
        

    def reset(self):
        self.env.reset()
        self.agent_ids = []
        d = {}
        for name in self.behavior_name:
            decision_steps, _ = self.env.get_steps(name)
            self.agent_ids.append(decision_steps.agent_id)
            d.update({name:decision_steps.obs[0]})
        self._num_agents = len(self.agent_ids)
        return d

    def step(self, actions):
        for name in actions.keys():
            self.env.set_actions(name, ActionTuple(continuous=actions[name]))


        self.env.step()

        d = {}
        for name in actions.keys():
            decision_steps, terminal_steps = self.env.get_steps(name)
            obs = []
            rewards = []
            dones = []
            for aid in self.agent_ids:
                aid = aid[0]
                if aid in decision_steps:
                    obs.append(decision_steps[aid].obs[0])
                    rewards.append(decision_steps[aid].reward)
                    dones.append(False)
                elif aid in terminal_steps:
                    obs.append(terminal_steps[aid].obs[0])
                    rewards.append(terminal_steps[aid].reward)
                    dones.append(True)
            infos = [{}]
            d.update({name:(np.array(obs), np.array(rewards), np.array(dones), infos)})

        return d

    def get_observation_space(self):
        d = {}
        for name in self.behavior_name:
            d.update({name:{"shape": self.spec[name].observation_specs[0].shape, "type": "visual"}})
        return d

    def get_action_space(self):
        d = {}
        for name in self.behavior_name:
            d.update({name:{"shape": self.spec[name].action_spec.continuous_size, "type": "continuous"}})
        return d


    

