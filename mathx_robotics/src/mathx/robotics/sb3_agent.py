import dm_env
import acme
from acme import types
from .agent import Agent

import gymnasium as gym
import gymnasium.spaces as gymsp
import stable_baselines3 as sb3
import stable_baselines3.common.noise as sb3noise
import stable_baselines3.common.logger as sb3log

import numpy as np
import copy

class Env(gym.Env):
  """
  https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
  
  This is used as a dummy environment to provide on observation and action space parameters to the agent. 
  """

  def __init__(self,state_space,action_space):
    def convert_space(space):
      return gymsp.Box(low=space.low,high=space.high,shape=(space.size()),dtype=np.float32)

    #self.observation_space=convert_space(state_space)
    self.observation_space=gymsp.Box(low=-float('inf'),high=float('inf'),shape=(state_space,),dtype=np.float32)
    self.action_space=convert_space(action_space)

  def step(self, action):
    raise NotImplementedError

  def reset(self, seed=None, options=None):
    raise NotImplementedError

  def render(self):
    raise NotImplementedError

  def close(self):
    raise NotImplementedError

class SB3Agent(Agent):
  """
  https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html#evaluate_policy
  https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
  """

  def __init__(self,params,state_space,action_space):
    self.state_space=state_space
    self.action_space=action_space
    self.env=Env(state_space,action_space)
    n_actions = self.env.action_space.shape[-1]
    self.action_noise = sb3noise.NormalActionNoise(mean=np.zeros(n_actions),sigma=.1 * np.ones(n_actions))
    self.agent=sb3.ddpg.DDPG(
      policy="MlpPolicy",
      gamma=params.get("beta",.99),
      env=self.env,
      action_noise=self.action_noise,
      verbose=1)
    self.agent.set_logger(sb3log.configure(None,["stdout"]))
    self.batch_size=params.get("batch_size",64)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    action,_=self.agent.predict(observation)
    #We need to apply noise here as SB3 only adds noise in their internal training loop.
    #TODO: Figure out a reasonable scale/parameterize it
    action=action+self.action_noise()*(self.action_space.high-self.action_space.low)
    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    self.state=copy.deepcopy(timestep.observation)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    #TODO: Populate replay before?
    #https://github.com/DLR-RM/stable-baselines3/blob/656de97269c9e3051d3bbfb3f5f328d486867bd8/stable_baselines3/common/off_policy_algorithm.py#L441
    #_store_transition
    #raise NotImplementedError
    state0=self.state
    action0=self.action_space.normalize(copy.deepcopy(action))
    state1=copy.deepcopy(next_timestep.observation)
    feedback=copy.deepcopy(next_timestep.reward)

    self.agent.replay_buffer.add(
      state0,
      state1,
      action0,
      feedback,
      np.array(0), #done, where (1-done) is a factor in the q1 term in the objective.
      [{}],
    )

    self.state=state1

  def update(self, wait: bool = False):
    self.agent.train(gradient_steps=1,batch_size=self.batch_size)
    return 0,0
  
  def save(self,path):
    log.info("WARNING: save not implemented")
  
  def restore(self,path):
    log.info("WARNING: restore not implemented")

  def q(self,x,a):
    return 0
