import optax
import flax.nnx as nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
# import jax.nn as jnn
# import time
from . import log
# from . import jax_utilities

"""
https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/sac/learning.py
https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/sac/networks.py
"""

# @nnx.jit
# def optimal_action_batch(policy,state):
#   optimal_action_vmap=jax.vmap(policy.optimal_action,in_axes=(0))
#   return optimal_action_vmap(state)
#   #return jnp.array(q.optimal_action(state[0]))

# jit is slower??? 40ms rather than 10ms, which is already very slow, and only 5ms if we stay in numpy array land
# @nnx.jit
def batch_to_arrays(batch):
  state0=np.concatenate([s[0][None,...] for s in batch])
  action0=np.concatenate([s[1][None,...] for s in batch])
  state1=np.concatenate([s[2][None,...] for s in batch])
  # action1=jnp.concatenate([s[3][None,...] for s in batch])
  feedback=np.concatenate([s[3][None,...] for s in batch])
  discount=np.concatenate([s[4][None,...] for s in batch])

  # return state0,action0,state1,action1,feedback
  return state0,action0,state1,feedback,discount

# # def q_objective(q,beta,state0,action0,state1,action1,feedback):
# def q_objective(q,policy,beta,stop_gradient,state0,action0,state1,feedback,discount):
#   action1=policy(state1)
#   q_value1=q(state1,action1)
#   new_value=feedback+beta*q_value1
#   if stop_gradient:
#     new_value=jax.lax.stop_gradient(new_value)

#   old_value=q(state0,action0)

#   return (new_value-old_value)**2

# def policy_objective(q,policy,state):
#   action=policy(state)
#   objective=-q(state,action)

#   return objective

def l2_objective(x, weight):
    return weight * (x ** 2).sum()

def l2_module(module, weight=0.001):
  return jnp.sum(
    [l2_objective(w, weight=weight)
    for w in jax.tree.leaves(module,is_leaf=lambda x:isinstance(x,nnx.Variable))]
  )

@nnx.jit(static_argnums=[6])
def update_disjoint(q_optimizer,policy_optimizer,q,policy,q_target,policy_target,hyperparams,batch):
  #####################################
  # state0,action0,state1,action1,feedback=batch
  state0,action0,state1,feedback,discount=batch

  #####################################
  # Compute gradients
  def q_objective(q,policy,q_target,policy_target,hyperparams,state0,action0,state1,feedback,discount):
    # action1=policy(state1)
    action1=policy_target(state1)

    q0=q(state0,action0)
    #TODO: Handle multiple q_target
    # q1=q(state1,action1)
    q1=q_target[0](state1,action1)

    # jax.debug.print("q0={q0} q1={q1} state0={s0} action0={a0}",
    #                 q0=q0.mean(),q1=q1.mean(),s0=state0.mean(),a0=action0.mean())

    q0_target=feedback+discount*hyperparams.beta*q1

    # error=q0-q0_target
    if not hyperparams.use_q_target_gradient:
      q0_target=jax.lax.stop_gradient(q0_target)

    error=q0-q0_target

    obj=0.
    obj+=1./2.*(error)**2

    #Over estimation penalty
    # Penalize feedback overestimates, f_est=q(s0,a0)-beta*q(s1,a1)
    if hyperparams.feedback_overestimate_penalty!=0:
      obj+=hyperparams.feedback_overestimate_penalty*jnn.relu(q0-q0_target)

    #Minimize random actions
    # obj+=jnp.abs(q0-(feedback+beta*q1))
    if hyperparams.cql!=0:
      obj+=hyperparams.cql*q(state0,policy(state0))
      obj+=hyperparams.cql*q(state0,action0)

    # actionr=policy.action_space.sample
    # Minimize over estimation of predict action vs recorded action
    if hyperparams.out_of_distribution_regularization!=0:
      obj+=hyperparams.out_of_distribution_regularization*jnn.relu(q(state0,policy(state0))-q(state0,action0))

    # Minimize over estimation of value at new state
    # obj+=jnn.relu(q(state1,action1)-q(state0,action0))

    #obj+=l2_module(q)
    
    # jax.debug.print("obj={obj}",obj=obj)

    return obj
  def q_objective_batch(q,policy,q_target,policy_target,hyperparams,state0,action0,state1,feedback,discount):
    objective_vmap=jax.vmap(q_objective,in_axes=(None,None,None,None,None,0,0,0,0,0))
    return jnp.sum(objective_vmap(q,policy,q_target,policy_target,hyperparams,state0,action0,state1,feedback,discount))
  q_objective_batch_grad=nnx.value_and_grad(q_objective_batch,argnums=0)
  q_obj,q_grad=q_objective_batch_grad(q,policy,q_target,policy_target,hyperparams,state0,action0,state1,feedback,discount)
  
  # jax.debug.print("q_obj={q_obj}",q_obj=q_obj)

  def policy_objective(q,policy,q_target,hyperparams,state):
    obj=0.
    # obj+=-alpha*q(state,policy(state))

    action_linear=policy(state,nonlinear=False)
    action=policy.action_space.apply_nonlinearity(action_linear)

    # Prevent actions from saturating
    # TODO: Incorporate this into action space
    obj+=hyperparams.action_regularization*jnp.sum(action_linear**2)

    # TODO: Handle multiple q_target
    q_value=q_target[0](state,action)
    # q_value=q(state,action)
    obj+=-hyperparams.policy_optimality*q_value
    # jax.debug.print("state={} action={} obj_q_value={}",state,action,q_value)

    #obj+=l2_module(policy)

    return obj
  def policy_objective_batch(q,policy,q_target,hyperparams,state):
    objective_vmap=jax.vmap(policy_objective,in_axes=(None,None,None,None,0))
    return jnp.sum(objective_vmap(q,policy,q_target,hyperparams,state))
  policy_objective_batch_grad=nnx.value_and_grad(policy_objective_batch,argnums=1)
  policy_obj,policy_grad=policy_objective_batch_grad(q,policy,q_target,hyperparams,state0)

  #####################################
  # Do update
  q_optimizer.update(q_grad)
  policy_optimizer.update(policy_grad)
  
  def polyak_update(x,y):
    return (1-hyperparams.target_learning_rate)*x+hyperparams.target_learning_rate*y

  # Update target networks
  q_state=nnx.state(q)
  for qt in q_target:
    qt_state=nnx.state(qt)
    qt_state_updated=jax.tree.map(polyak_update,qt_state,q_state)
    nnx.update(qt,qt_state_updated)

  policy_target_state=nnx.state(policy_target)
  policy_state=nnx.state(policy)
  policy_target_state_updated=jax.tree.map(polyak_update,policy_target_state,policy_state)
  nnx.update(policy_target,policy_target_state_updated)

  #####################################
  # print("sample",state0,jnp.argmax(action0),state1,jnp.argmax(action1_),feedback,objective)
  return q_obj,policy_obj

# @nnx.jit
# def update_joint(q_optimizer,policy_optimizer,q,policy,beta,batch):
#   #####################################
#   state0,action0,state1,feedback=batch

#   #####################################
#   # Compute gradients
#   def joint_objective(q,policy,beta,state0,action0,state1,feedback):
#     policy_weight=.01
#     q_obj=q_objective(q,policy,beta,False,state0,action0,state1,feedback)
#     policy_obj=policy_objective(q,policy,state0)
#     return q_obj + policy_weight*policy_obj,q_obj,policy_obj
#   def joint_objective_batch(q,policy,beta,state0,action0,state1,feedback):
#     objective_vmap=jax.vmap(joint_objective,in_axes=(None,None,None,0,0,0,0),out_axes=(0,0,0))
#     obj,q_obj,policy_obj=objective_vmap(q,policy,beta,state0,action0,state1,feedback)
#     return jnp.sum(obj),(jnp.sum(q_obj),jnp.sum(policy_obj))
#   joint_objective_batch_grad=nnx.value_and_grad(joint_objective_batch,argnums=(0,1),has_aux=True)
#   (objective,(q_obj,policy_obj)),(q_grad,policy_grad)=joint_objective_batch_grad(q,policy,beta,state0,action0,state1,feedback)

#   #####################################
#   # Do update
#   q_optimizer.update(q_grad)
#   policy_optimizer.update(policy_grad)

#   #####################################
#   return q_obj,policy_obj

# @nnx.jit
# def update_joint(optimizer,q,policy,beta,batch):
#   pass

class Optimizer(nnx.Module):
  def __init__(self,params,q,policy):
    self.params=params #Hyperparameters(**params)

    self.q=q
    self.policy=policy

    self.num_q_target=1

    # TODO: Seed each clone separately
    self.q_target=[nnx.clone(self.q) for _ in range(self.num_q_target)]
    self.policy_target=nnx.clone(self.policy)
    # self.q_target=[]
    # self.policy_target=None

    self.q_optimizer=nnx.Optimizer(
      self.q,
      optax.adamw(learning_rate=self.params.q_learning_rate,
                  b1=self.params.momentum,
                  weight_decay=self.params.weight_decay))
    self.policy_optimizer=nnx.Optimizer(
      self.policy,
      optax.adamw(learning_rate=self.params.q_learning_rate,
                  b1=self.params.momentum,
                  weight_decay=self.params.weight_decay))

  def update(self,batch):
    # Zip batch elements
    # log.info("converting to arrays")
    batch_arrays=batch_to_arrays(batch)

    # Apply update
    # log.info("update")
    # log.info("BLAH"*10)
    # log.info(f"q={self.q(batch[0][0],batch[0][1])}")
    q_objective,policy_objective=update_disjoint(self.q_optimizer,self.policy_optimizer,self.q,self.policy,self.q_target,self.policy_target,self.params,batch_arrays)
    # q_objective,policy_objective=update_joint(self.q_optimizer,self.policy_optimizer,self.q,self.policy,self.beta,batch_arrays)
    # import sys; sys.exit(0)
    
    # print(f"q_objective={q_objective} BLUE")

    batch_size=len(batch)
    return q_objective/batch_size,policy_objective/batch_size

def epsilon_greedy_action(x,epsilon,action_space,rand,policy):
  if(rand.uniform(0,1)>=epsilon):
    action=policy.optimal_action(x)
  else:
    #action=np.zeros(policy.action_space)
    #action[rand.randint(0,policy.action_space-1)]=1
    action=action_space.to_real(action_space.sample(rand))
  return action
