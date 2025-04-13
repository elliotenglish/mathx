import numpy as np
# import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jax
import jax.core
import jax.scipy.optimize as jspo
import scipy.optimize as spo
from . import mlp
from . import utilities

class QFunctionComposite(nnx.Module):
  def __init__(self,params):
    self.state_space=params["state_space"]
    self.action_space=params["action_space"]
    self.model=mlp.MLP({"layers":params["layers"]+[{"out_features":1,"activation":None}],
                        "in_features":self.state_space+self.action_space.real_size()})

  @nnx.jit
  def __call__(self,x,a):
    return self.model(jnp.concat([x,a]))[0]

  @nnx.jit
  def optimal_action(self,x,a_guess=None,num_steps=200,learning_rate=1e-1,momentum=0):
    # print("optimal_action")

    def objective(a):
      return -self(x,self.action_space.apply_nonlinearity(a))

    value_and_grad=jax.value_and_grad(objective)

    if a_guess is not None:
      a_linear=a_guess
    else:
      a_linear=jnp.zeros([self.action_space.real_size()])

    res=jspo.minimize(objective,x0=a_linear,method="BFGS")

    # res=spo.minimize(value_and_grad,Jax=True,x0=a_linear,method="BFGS")

    # if not isinstance(x,jax.core.Tracer):
    #   print("scipy optimizer",res.x,objective(res.x),res.nit)

    # return res.x

    # _,mom=value_and_grad(a_linear)
    # for i in range(num_steps):
    #   # a_linear=a_linear
    #   obj,grad=value_and_grad(a_linear)

    #   mom=momentum*mom+grad
    #   a_linear=a_linear-learning_rate*mom
    #   # if not isinstance(x,jax.core.Tracer):
    #   #   print(f"step={i: 5d} objective={obj: 16g} gradient_norm={jnp.linalg.norm(grad): 16g} action={self.action_space.apply_nonlinearity(a_linear)}")

    # a_sol=self.action_space.apply_nonlinearity(a_linear)
    # print("internal optimizer",a_sol,objective(a_sol))

    return self.action_space.apply_nonlinearity(res.x)

class QFunctionImplicit(nnx.Module):
  def __init__(self,params):
    self.state_space=params["state_space"]
    self.action_space=params["action_space"]
    self.model=mlp.MLP({"layers":params["layers"]+[{"out_features":1,"activation":None}],
                        "in_features":self.state_space+self.action_space})
    #The final layer must be linear with no activation in order to model arbitrary value functions

  def __call__(self,x,a):
    return self.model(jnp.concat([x,a]))[0]

  def optimal_action(self,x):
    # print(x)
    values=jnp.concat([self(x,utilities.one_hot_vector(self.action_space,i)) for i in range(self.action_space)])
    # values=utilities.one_hot_vector(self.action_space,0)
    #values=jnp.concat([self(x,utilities.one_hot_vector(5,i)) for i in 5])
    max_idx=jnp.argmax(values)
    # print(f"max_idx={max_idx} values={values}")
    return utilities.one_hot_vector(self.action_space,max_idx)

class QFunctionExplicit(nnx.Module):
  def __init__(self,params):
    self.state_space=params["state_space"]
    self.action_space=params["action_space"]
    self.model=mlp.MLP({"layers":params["layers"]+[{"out_features":self.action_space,"activation":None}],
                        "in_features":self.state_space})
    #The final layer must be linear with no activation in order to model arbitrary value functions

  def __call__(self,x,a):
    max_idx=np.argmax(a)
    return self.model(x)[max_idx]

  def optimal_action(self,x):
    values=self.model(x)
    max_idx=np.argmax(values)
    # print(f"max_idx={max_idx} values={values}")
    return utilities.one_hot_vector(self.action_space,max_idx)
