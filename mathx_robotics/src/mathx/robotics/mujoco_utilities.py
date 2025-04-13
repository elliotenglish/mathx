import mujoco
import mujoco.viewer
# import cv2

renderer=None
scene_option=None
viewer=None

def print_stats(model):
  """
  https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel
  model.body(i) [model.nbody] corresponds to the "bodies" section
  model.joint(i) [model.njnt, model.nv] corresponds to the "joints" and "dofs" sections
  model.actuator(i) [model.nu] corresponds to the "actuators" section
  """

  print("bodies")
  for i in range(model.nbody):
    body=model.body(i)
    print(f"id={i} name={body.name} mass={body.mass} inertia={body.inertia}")

  print("joints")
  for i in range(model.njnt):
    jnt=model.joint(i)
    print(f"id={i} name={jnt.name} type={jnt.type} qposadr={jnt.qposadr} M0={jnt.M0} range={jnt.range if jnt.limited else None}")

  print("actuators")
  for i in range(model.nu):
    act=model.actuator(i)
    print(f"id={i} name={act.name} ctrlrange={act.ctrlrange} trnid={act.trnid} actadr={act.actadr} actnum={act.actnum}")

def visualize(model,data,timeout=0):
  global renderer
  global scene_option

  # if renderer is None or renderer.model!=model:
  #   print("allocating renderer")
  #   renderer=mujoco.Renderer(model)
  #   scene_option = mujoco.MjvOption()
  #   scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

  # renderer.update_scene(data, scene_option=scene_option)
  # pixels = renderer.render()
  # cv2.imshow("scene",pixels)
  # cv2.waitKey(timeout)

  global viewer

  if viewer is None:
    viewer=mujoco.viewer.launch_passive(model, data, show_right_ui=False)
  viewer.sync()
