import gymnasium as gym
import mathx.learning.system as cls
import mathx.learning.space as clsp
import jax.numpy as jnp
from test_config import debug_visualize

class GymnasiumSystem(cls.System):
  """
  "Pendulum-v1"
  """
  def __init__(self,example_name,**kwparms):
    self.env = gym.make(example_name,render_mode="human" if debug_visualize() else None,**kwparms)

    # self.goal_pos=math.pi*.5
    self.num_actions=9

  def reset(self,rand):
    self.internal_state=self.env.reset()[0]
    self.internal_done=False
    self.internal_reward=None

  def state(self):
    return jnp.array(self.internal_state)

  def transition(self,action):
    if isinstance(self.env.action_space,gym.spaces.Discrete):
      env_action=action[0]
      # print(action,env_action)
    else:
      env_action=action
      # action=[self.env.action_space.low[0]+float(action_idx)/(self.num_actions-1)*(self.env.action_space.high[0]-self.env.action_space.low[0])]
    self.internal_state,self.internal_reward,self.internal_done,truncated,_=self.env.step(env_action)

  def feedback(self):
    return jnp.array(self.internal_reward)

  def state_space(self):
    return self.env.observation_space.shape[0]

  def action_space(self):
    if isinstance(self.env.action_space,gym.spaces.Discrete):
      return clsp.DiscreteSpace(1,self.env.action_space.n)
    else:
      return clsp.ContinuousSpace(self.env.action_space.shape[0],self.env.action_space.low,self.env.action_space.high)

  def done(self):
    return self.internal_done
