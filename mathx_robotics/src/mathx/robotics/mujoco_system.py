from . import system
from . import space
from . import mujoco_utilities
from . import utilities

import mujoco
import numpy as np
import math
import jax.numpy as jnp

class MujocoSystem(system.System):
  def __init__(self,xml_string=None,xml_path=None,feedback_function=None,initialize_function=None,**params):
    self.visualize=params.get("visualize",False)
    self.feedback_function=feedback_function
    self.initialize_function=initialize_function

    if xml_path is not None:
      self.model=mujoco.MjModel.from_xml_path(xml_path)
    elif xml_string is not None:
      self.model=mujoco.MjModel.from_xml_string(xml_string)
    else:
      assert False

    # Hardcode integrator to implicit
    self.model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT

    self.data=mujoco.MjData(self.model)
    mujoco_utilities.print_stats(self.model)

    for i in range(self.model.nu):
      assert self.model.actuator(i).ctrlrange[1]>self.model.actuator(i).ctrlrange[0]

    self.action_space_=space.ContinuousSpace(
      size=self.model.nu,
      low=[self.model.actuator(i).ctrlrange[0] for i in range(self.model.nu)],
      high=[self.model.actuator(i).ctrlrange[1] for i in range(self.model.nu)])

    self.velocity_maximum=24*math.pi
    self.termination_feedback=-10

    # Compute actuator limits
    self.reset(None)

    if self.visualize:
      mujoco_utilities.visualize(self.model,self.data,1)

  def state_space(self):
    return self.state().shape[0]

  def action_space(self):
    return self.action_space_

  def state(self):
    qpos=np.copy(self.data.qpos)
    for i in range(self.model.njnt):
      jnt=self.model.joint(i)
      if jnt.type==3 and not jnt.limited:
        qpos[jnt.qposadr]=utilities.modulus_zero(qpos[jnt.qposadr],2*math.pi)
    return np.concatenate([qpos,self.data.qvel])

  def transition(self,action):
    if action is not None:
      self.data.ctrl=action

    mujoco.mj_step(self.model,self.data)

    if self.visualize:
      mujoco_utilities.visualize(self.model,self.data,1)

    # print(self.state())

  def feedback(self):
    if self.feedback_function:
      f=self.feedback_function(self.model,self.data)
      if self.done():
        f+=self.termination_feedback
      return f
    raise NotImplementedError

  def done(self):
    # This is to capture the simulation becoming unstable
    return np.abs(self.data.qvel).max()>self.velocity_maximum

  def reset(self,rand):
    mujoco.mj_resetData(self.model,self.data)
    if self.initialize_function is not None:
      self.initialize_function(self.model,self.data,rand)
