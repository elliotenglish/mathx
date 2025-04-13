# import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from . import mlp
# from . import utilities
from . import jax_utilities

class PolicyFunction(nnx.Module):
  def __init__(self,params):
    self.state_space=params["state_space"]
    self.action_space=params["action_space"]
    self.model=mlp.MLP({"layers":params["layers"]+[{"out_features":self.action_space.real_size(),"activation":None}],
                        "in_features":self.state_space})

  def __call__(self,x,nonlinear=True):
    a_linear=self.model(x)

    if nonlinear:
      a=self.action_space.apply_nonlinearity(a_linear)
    else:
      a=a_linear

    # if not jax_utilities.in_jax_jit(x):
    #   print(f"state={x} action={a} action_linear={a_linear}")

    return a

  def optimal_action(self,x,nonlinear=True):
    return self(x,nonlinear=nonlinear)
