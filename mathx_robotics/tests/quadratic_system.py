import mathx.robotics.system as cls
import mathx.robotics.space as clsp
import jax.numpy as jnp
import jax
import random
# import math
import numpy as np

class QuadraticSystem(cls.System):
  """
  This represents an optimization problem through the interface of a reinforcement learning problem.
  state - get the current state
  feedback - get the feedback for the current state and
  """

  def __init__(self,dimensions,discrete_action):
    import fiblat
    self.dimensions=dimensions
    if self.dimensions>=2:
      self.num_angles=5**self.dimensions
      self.angles=fiblat.sphere_lattice(self.dimensions,self.num_angles)
    else:
      self.num_angles=2
      self.angles=np.array([[-1],[1]])
    self.num_octaves=6
    self.step_min=1e-3
    self.discrete_action=discrete_action

    self.reset(np.random.default_rng(543223))

  def reset(self,rand):
    # self.center=rand.uniform(low=-1,high=1,size=[self.dimensions])
    self.center=rand.uniform(low=0,high=0,size=[self.dimensions])
    self.hidden_state=rand.uniform(low=-1,high=1,size=[self.dimensions])

  def transition(self,action):
    if(self.discrete_action):
      action_idx=action[0]
      # action_idx=jnp.argmax(action)
      angle=int(action_idx/self.num_octaves)
      step=self.step_min*(2**int(action_idx%self.num_octaves))
      self.hidden_state+=step*self.angles[angle]
      # print(f"transition action_idx={action_idx} angle={angle} step={step}")
    else:
      self.hidden_state+=action
      # print(f"transition action={action}")

  def state(self):
    return jnp.concat([self.hidden_state,self.objective_and_gradient()[1]])

  def feedback(self):
    return -self.objective_and_gradient()[0]

  def objective_and_gradient(self):
    def objfn(x):
      dx=x-self.center
      return jnp.matmul(dx.T,dx)
    objgradfn=jax.value_and_grad(objfn)
    obj,grad=objgradfn(self.hidden_state)
    # print(f"obj={obj} grad={grad}")
    return obj[None],grad

  def state_space(self):
    return self.state().shape[0]

  def action_space(self):
    if self.discrete_action:
      return clsp.DiscreteSpace(1,self.num_angles*self.num_octaves)
    else:
      return clsp.ContinuousSpace(self.dimensions,-1,1)
