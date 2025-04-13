from . import utilities
from . import checkpoint_utilities

# import sys
import os
# import random
import natsort
import glob
import pickle
import numpy as np
# import orbax.checkpoint
import flax.nnx as nnx
# import flax.training.orbax_utils
from . import log
import shutil
import pprint
import dm_env
import time

class Solver:
  def __init__(self,
               system,
               agent,
               **params):
    self.system=system
    self.agent=agent
    self.max_episodes=params.get("max_episodes",None)
    self.max_steps=params.get("max_steps",100)
    self.exit_on_done=params.get("exit_on_done",True)
    self.update=params.get("update",True)

    self.rand=np.random.default_rng(45234)
    self.output_dir=params.get("output_dir")
    os.makedirs(self.output_dir,exist_ok=True)

    self.checkpoint_dir=os.path.join(os.path.abspath(self.output_dir),"checkpoints")
    self.checkpoint_period=params.get("checkpoint_period",None)
    self.checkpoint_enabled=self.checkpoint_period is not None

    self.log_file=os.path.join(self.output_dir,"log.txt")

    if not self.checkpoint_enabled:
      if os.path.exists(self.checkpoint_dir): shutil.rmtree(self.checkpoint_dir)

    # ckpt_mgr_options = orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=3, keep_period=2)
    # self.ckpt_mgr=orbax.checkpoint.CheckpointManager(
    #   self.checkpoint_dir,options=ckpt_mgr_options)

    log.initialize(self.log_file,append=self.checkpoint_enabled)

    log.info(pprint.pformat(self.__dict__))

  def solve(self):
    self.episode_idx=0

    if self.checkpoint_enabled:
      ckpts=checkpoint_utilities.checkpoint_list(self.checkpoint_dir) 
      
      if(len(ckpts)>0):
        self.episode_idx=ckpts[-1]
        self.agent.restore(checkpoint_utilities.checkpoint_path(self.checkpoint_dir,self.episode_idx))
      else:
        self.agent.save(checkpoint_utilities.checkpoint_path(self.checkpoint_dir,self.episode_idx))

    while (self.max_episodes is None or self.episode_idx<self.max_episodes):
      # log.info("outer loop")
      #problem=SingleJointProblem()
      self.system.reset(self.rand)

      episode_feedback=0
      # episode_q_objective=0
      # episode_policy_objective=0

      step_idx=0

      timestep=dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=self.system.state()
      )
      self.agent.observe_first(timestep)

      while (self.max_steps is None or step_idx<self.max_steps) and (not self.exit_on_done or not self.system.done()):
        t0=time.time()

        ##############################
        # Evaluate policy and transition
        # log.info("get state")
        state0=self.system.state()

        # Select action, sampling is done internally by the agent
        # log.info("get action")
        action0=self.agent.select_action(state0)
        if np.isnan(action0).any():
          log.info(f"state0={state0} action0={action0}")
          print(self.agent.q)
          print(self.agent.policy)
          import pdb
          pdb.set_trace()

        # Step the system
        # log.info("transition")
        self.system.transition(self.system.action_space().from_real(action0))
        feedback=self.system.feedback()
        state1=self.system.state()
        done=self.system.done()# or (self.max_steps is not None and step_idx==(self.max_steps-1))

        ##############################
        # Pass state update, action and feedback to agent
        # log.info("observe")
        timestep=dm_env.TimeStep(
          step_type=dm_env.StepType.MID,
          reward=feedback,
          discount=np.array(0) if done else np.array(1),
          observation=state1
        )

        self.agent.observe(action0,timestep)

        # Do update
        # log.info("update")
        if self.update:
          objectives=self.agent.update(wait=True)

        # Accumulate objectives
        episode_feedback+=feedback
        # episode_q_objective+=q_objective
        # episode_policy_objective+=policy_objective

        ###############
        # Logging

        t1=time.time()

        # log.info(batch)
        # log.info(f"debug batch_q_error={batch_q_error/self.batch_size} q_error={q_error}")
        log.info(f"step_idx={step_idx: 5d} step_feedback={feedback} step_done={done} state0={state0[:3]} action0={action0[:3]} steps_s={1/(t1-t0)} q_objective={objectives[0]} policy_objective={objectives[1]}")# step_q_objective={q_objective/self.batch_size: 16g} step_policy_objective={policy_objective/self.batch_size: 16g} q_value0={q_value0} q_value1={q_value1} q_error={q_error}")

        ###############
        # Increment step
        step_idx+=1

      log.info(f"episode_idx={self.episode_idx} episode_steps={step_idx} episode_feedback={episode_feedback}")# episode_q_objective={episode_q_objective/(step_idx*self.batch_size)} episode_policy_objective={episode_policy_objective/(step_idx*self.batch_size)}")
      self.episode_idx+=1
      if self.checkpoint_enabled and (self.episode_idx%self.checkpoint_period)==0:
        self.agent.save(checkpoint_utilities.checkpoint_path(self.checkpoint_dir,self.episode_idx))
