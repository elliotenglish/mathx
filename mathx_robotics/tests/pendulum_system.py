import mujoco
import jax.numpy as jnp
import numpy as np
from mathx.robotics.mujoco_system import MujocoSystem

def GenerateXML(num_bodies,length=.4,body_prefix="body_",marker_name="marker_ee_pos"):
  body_xml=""
  actuator_xml=""
  for i in range(num_bodies):
    body_xml+=f"""<body name="{body_prefix}{i}" pos="0 0 {num_bodies*length+.1 if i==0 else length}" euler="0 0 0">
  <joint name="{body_prefix}joint_{i}" type="hinge" axis="1 0 0" pos="0 0 0"/>
  <geom name="{body_prefix}box_{i}" type="box" pos="0 0 {length*.5}" size=".02 .02 {length*.5}" rgba="1 0 0 1"/>
"""
    if i==(num_bodies)-1:
      body_xml+=f"""<site type="sphere" name="{marker_name}" pos="0 0 {length}" size="0.06" rgba="0 1 0 0.2" />
"""
  for i in range(num_bodies):
    body_xml+="""</body>
"""

  mujoco_config = f"""<mujoco>
  <option gravity="0 0 -10" viscosity=".1" timestep=".01"/>
  <worldbody>
    {body_xml}

    <light name="top" pos="0 0 3"/>
    <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3" />
  </worldbody>
  <actuator>
    <general name="joint_0_actuator" dynprm="1 0 0" gainprm="1 0 0" biasprm="0 0 0" joint="{body_prefix}joint_0" ctrlrange="-5 5" />
    <!-- <general name="joint_1_actuator" dynprm="1 0 0" gainprm="1 0 0" biasprm="0 0 0" joint="{body_prefix}joint_1" ctrlrange="-1 1" /> /-->
  </actuator>

  <asset>
    <!-- <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32"
            height="512" /> -->
    <!-- <texture name="body" type="cube" builtin="flat" mark="cross" width="128" height="128"
            rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" /> -->
    <!-- <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1" /> -->
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3"
      rgb2=".2 .3 .4" />
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2" />
  </asset>

  <visual>
    <global azimuth="45" elevation="-45" />
  </visual>
</mujoco>
  """

  print(mujoco_config)
  return mujoco_config

def PendulumSystem(num_bodies,visualize):
  def feedback(model,data):
    feedback=0

    # Minimize velocity
    feedback+=-.001*jnp.sum(data.qvel**2)

    # TODO: Minimize energy usage, should be by d/dt(f*dx) = df/dt * dx + f*dx/dt => f*vel
    # For now just minimize force
    # feedback+=-.001*jnp.sum(data.ctrl**2)
    # feedback+=-.001*jnp.sum

    # Task goal
    marker_ee_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'marker_ee_pos')
    pos=data.site_xpos[marker_ee_id]
    feedback+=pos[2]

    return feedback
  
  def initialize(model,data,rand):
    if rand is not None:
      data.qpos=rand.uniform(-.2*np.pi,.2*np.pi,size=[num_bodies])
      data.qpos[0]=data.qpos[0]+np.pi

  mujoco_config=GenerateXML(num_bodies)
  return MujocoSystem(xml_string=mujoco_config,
                      feedback_function=feedback,
                      initialize_function=initialize,
                      visualize=visualize)
