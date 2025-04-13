from mathx.learning.system import System
from mathx.learning.custom_agent import CustomAgent
# from mathx.learning.acme_agent import AcmeAgent
from mathx.learning.sb3_agent import SB3Agent
from mathx.learning.space import ContinuousSpace
from mathx.learning.solver import Solver
import mathx.learning.geometry as clgeom
from test_config import *
import numpy as np
import math

class ConstantSystem(System):
  def __init__(self,beta):
    self.ns=2
    self.na=self.ns
    self.beta=beta

  def state_space(self):
    return self.ns

  def action_space(self):
    return ContinuousSpace(self.na,-2,3)

  def reset(self,rand):
    self.x=rand.uniform(low=-3,high=3,size=[self.ns])
    # print(f"x={self.x} optimal_q={self.q_value_analytic(self.x)}")

  def state(self):
    return np.copy(self.x)

  def transition(self,action):
    self.x[:self.na]=np.clip(self.x[:self.na]+action,-10,10)
    # pass

  def feedback_fn(self,x):
    return -.5*np.linalg.norm(x) + .33
    # return np.array(.33)

  def feedback(self):
    return self.feedback_fn(self.x)

  def action_optimal(self,x):
    dist=np.linalg.norm(x)
    if dist==0:
      return np.array([0.]*self.na)

    dir=-x/dist
    _,_,max_step=clgeom.ray_intersect_aabb(x,dir,self.action_space().low,self.action_space().high)

    return min(dist,max_step)*dir

  def q_value_analytic(self,x):
    beta_prod=1
    q=0
    while beta_prod>1e-3:
      x=x+self.action_optimal(x)
      f=self.feedback_fn(x)
      q+=beta_prod*f
      beta_prod*=self.beta
      print(x,f,q,beta_prod)
    return q

  def done(self):
    return False

def helper_test_rl_convergence(agent_factory):
  beta=.9
  epsilon=.1
  decay_threshold=1e-3
  max_steps=2*math.log(decay_threshold)/math.log(beta)
  max_episodes=100

  system=ConstantSystem(beta=beta)
  agent=agent_factory({"beta":.9},system.state_space(),system.action_space())
  Solver(system,
         agent,
         max_steps=max_steps,
         max_episodes=max_episodes,
         output_dir="convergence",
         checkpoint_period=1 if debug_write() else None
         ).solve()

def test_rl_convergence_custom():
  def agent_factory(params,state_space,action_space):
    return CustomAgent(
      params=params,
      state_space=state_space,
      action_space=action_space)

  helper_test_rl_convergence(agent_factory=agent_factory)

# def test_rl_convergence_acme():
#   def agent_factory(state_space,action_space):
#     return AcmeAgent(
#       state_space=state_space,
#       action_space=action_space)

#   helper_test_rl_convergence(agent_factory=agent_factory)

def test_rl_convergence_sb3():
  def agent_factory(params,state_space,action_space):
    return SB3Agent(
      params,
      state_space=state_space,
      action_space=action_space)

  helper_test_rl_convergence(agent_factory=agent_factory)
