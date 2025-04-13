from . import q_function
from . import policy_function
from . import optimizer
from . import database
from . import models
from . import log
from . import jax_utilities
from .hyperparameters import Hyperparameters

from .agent import Agent

import os
import dm_env
# import acme
from acme import types
import numpy as np
import copy
import math
import flax.nnx as nnx
import orbax.checkpoint as ocp

class CustomAgent(Agent):
  """
  For documentation on dm_env.Timestep see:
  https://github.com/google-deepmind/dm_env/blob/91b46797fea731f80eab8cd2c8352a0674141d89/dm_env/_environment.py#L25
  """
  def __init__(self,params,state_space,action_space):
    self.epsilon=params.get("epsilon",.1)
    self.sigma=params.get("sigma",.2)
    self.alpha=params.get("alpha",0)
    self.state_space=state_space
    self.action_space=action_space
    random_seed=params.get("random_seed",5432453)
    # self.rand=np.random.default_rng(random_seed)
    self.rand=jax_utilities.Generator(random_seed)

    model_params={
      **models.architecture_params,
      "state_space":self.state_space,
      "action_space":self.action_space
    }

    # if params.get("explicit_q",True):
    #   self.q=q_function.QFunctionExplicit(model_params)
    # else:
    #   self.q=q_function.QFunctionImplicit(model_params)
    self.q_fn=q_function.QFunctionComposite(model_params)
    self.policy_fn=policy_function.PolicyFunction(model_params)

    # optimizer_params={
    #   "learning_rate":1e-4,
    #   "weight_decay":1e-4,
    #   "momentum":.9,
    #   "alpha":self.alpha,
    #   "beta":self.beta
    # }
    self.optimizer=optimizer.Optimizer(Hyperparameters.get_default(beta=params["beta"]),self.q_fn,self.policy_fn)

    self.database=database.Database(params.get("database",{}))
    self.batch_size=params.get("batch_size",64)
    # self.batch_size=1

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # return epsilon_greedy.epsilon_greedy_action(observation,self.epsilon,self.action_space,self.rand,policy=self.policy)
    return self.action_space.sample(self.rand,mean=self.policy_fn(observation),stddev=self.sigma)

  def observe_first(self, timestep: dm_env.TimeStep):
    self.state=timestep.observation

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    # Record observation
    state1=copy.deepcopy(next_timestep.observation)
    feedback=copy.deepcopy(next_timestep.reward)
    discount=next_timestep.discount
    data_point=(self.state,action,state1,feedback,discount)
    self.database.add(data_point)

    # Debugging output
    debug_output=False
    if debug_output:
      q_value0=self.q_fn(self.state,action)
      action1=self.policy_fn.optimal_action(state1)
      q_value1=self.q_fn(state1,action1)
      o=q_value0-(feedback+self.optimizer.hyperparameters.beta*q_value1)
      q_error=o*o
      log.info(f"q_value0={q_value0} beta={self.optimizer.hyperparameters.beta} q_value1={q_value1} o={o}")

    # Update current state
    self.state=copy.deepcopy(next_timestep.observation)

  def update(self, wait: bool = False):
    # Get training batch
    # log.info("sampling replay buffer")
    batch=self.database.sample(self.batch_size,self.rand)

    # Validate properties of batch
    # feedback_values=set([float(x[3][0]) for x in batch])
    # log.info(f"feedback_values={feedback_values}")

    # log.info("verifying stats")
    batch_stats=set([tuple([f.shape for f in x]+[f.dtype for f in x]) for x in batch])
    assert len(batch_stats)==1,str(batch_stats)
    # log.info(len(batch),batch_stats)

    ##############################
    # Compute Q error statistics

    debug_output=False
    if debug_output:
      # batch_q_error=0
      # for p in batch:
      #   s0,a0,s1,f=p
      #   old_value=self.q(s0,a0)
      #   a1=self.policy.optimal_action(s1)
      #   q_value1=self.q(s1,a1)
      #   new_value=f+self.beta*q_value1
      #   o=(new_value-old_value)**2
      #   # log.info(f"{id(self.policy)}")
      #   # log.info(f"solver q_objective {old_value} {new_value} {q_value1} {self.beta} {s0.sum()} {a0.sum()} {s1.sum()} {a1.sum()} {f}")
      #   batch_q_error+=o

      # log.info(f"solver state0={state0} action0={action0} state1={state1} action1={action1} feedback={feedback}")
      pass
    else:
      q_value0=0
      q_value1=0
      q_error=0

    ##############################
    # Do model update

    # log.info("optimizer")
    q_objective,policy_objective=self.optimizer.update(batch)

    # log.info(f"q_objective={q_objective} policy_objective={policy_objective}")

    # Error out on large objectives
    if q_objective>1e6 or policy_objective>1e6 or math.isnan(q_objective) or math.isnan(policy_objective):
      log.info("objective too large")
      log.info(f"q_objective={q_objective} policy_objective={policy_objective}")
      log.info("batch")
      log.info(batch)
      import pdb
      pdb.set_trace()

    return q_objective,policy_objective

  def get_state(self):
    return {
      "q":nnx.state(self.q_fn),
      "policy":nnx.state(self.policy_fn),
      "optimizer":nnx.state(self.optimizer)
    }

  def set_state(self,state):
    nnx.update(self.q_fn,state["q"])
    nnx.update(self.policy_fn,state["policy"])
    nnx.update(self.optimizer,state["optimizer"])

  def model_path(self,path):
    return os.path.join(path,"model")

  def db_path(self,path):
    return os.path.join(path,"database.pkl")

  def save(self,path):
    assert path[-1]!="/"

    # Write to a temporary directory first to avoid half writes
    tmp_path=path+".tmp"
    os.makedirs(tmp_path,exist_ok=True)
    ocp.test_utils.erase_and_create_empty(tmp_path)

    with ocp.StandardCheckpointer() as ckptr:
      ckptr.save(self.model_path(tmp_path),self.get_state())

    self.database.save(self.db_path(tmp_path))

    # Move into final place
    os.replace(tmp_path,path)

  def restore(self,path):
    if os.path.exists(self.model_path(path)):
      with ocp.StandardCheckpointer() as ckptr:
        st=ckptr.restore(self.model_path(path),self.get_state())
        self.set_state(st)
        # print(res)
        # import sys
        # sys.exit(-1)

    if os.path.exists(self.db_path(path)):
      self.database.restore(self.db_path(path))

  def q(self,x,a):
    return self.q_fn(x,a)
