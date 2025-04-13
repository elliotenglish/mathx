import pendulum_system
import gymnasium_system
import quadratic_system
from mathx.learning.custom_agent import CustomAgent
# from mathx.learning.acme_agent import AcmeAgent
from mathx.learning.sb3_agent import SB3Agent

from mathx.learning.solver import Solver
from mathx.learning.mujoco_system import MujocoSystem
from mathx.learning import log
from test_config import *
import numpy as np

# For each test we want to check the following:
# - q objective reduction
# - policy objective
# - episode feedback increase
# - time to done decreases
#
# analysis of log stats:
# - modify solver to store stats
# - add tool for running stats analysis
#   - compare values in sliding windows at beginning and end
#   - compare values in sliding windows with fixed separation

#Hack to get some initial prints before we get a log file path. TODO: refactor run_test to instantiate everything in callbacks and set the log before.
log.initialize()

def run_test(name,system,agent,default_steps,default_episodes):
  Solver(system,
         agent,
         max_steps=debug_value("TEST_STEPS",default_steps),
         max_episodes=debug_value("TEST_EPISODES",default_episodes),
         output_dir=name,
         checkpoint_period=50 if debug_write() else None).solve()

def test_reinforcement_learning_quadratic1d():
  system=quadratic_system.QuadraticSystem(1,False)
  agent=SB3Agent({"beta":.9},system.state_space(),system.action_space())
  run_test("optimization_quadratic1d",system,agent,50,200)

def test_reinforcement_learning_quadratic2d():
  system=quadratic_system.QuadraticSystem(2,False)
  agent=SB3Agent({"beta":.9},system.state_space(),system.action_space())
  run_test("optimization_quadratic2d",system,agent,50,200)

def test_reinforcement_learning_pendulum_mujoco():
  system=pendulum_system.PendulumSystem(1,visualize=debug_visualize())
  # agent=SB3Agent({"beta":.9},system.state_space(),system.action_space())
  agent=CustomAgent({"beta":.9},system.state_space(),system.action_space())
  run_test("pendulum_mujoco",system,agent,200,200)

def test_reinforcement_learning_pendulum_gymnasium():
  system=gymnasium_system.GymnasiumSystem("Pendulum-v1")
  # agent=SB3Agent({"beta":.9},system.state_space(),system.action_space())
  agent=CustomAgent({"beta":.9},system.state_space(),system.action_space())
  run_test("pendulum_gymnasium",system,agent,200,200)

def test_reinforcement_learning_acrobot():
  system=gymnasium_system.GymnasiumSystem("Acrobot-v1")
  agent=CustomAgent({"beta":.95},system.state_space(),system.action_space())
  run_test("acrobot",system,agent,500,100)

def test_reinforcement_learning_cart_pole():
  system=gymnasium_system.GymnasiumSystem("CartPole-v1",sutton_barto_reward=True)
  agent=CustomAgent({"beta":.95},system.state_space(),system.action_space())
  run_test("cart_pole",system,agent,500,100)

def test_reinforcement_learning_inverted_pendulum():
  """
  https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
  """
  system=gymnasium_system.GymnasiumSystem("InvertedPendulum-v5")
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("inverted_pendulum",system,agent,500,100)

def test_reinforcement_learning_inverted_double_pendulum():
  """
  https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/
  """
  system=gymnasium_system.GymnasiumSystem("InvertedDoublePendulum-v5")
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("inverted_double_pendulum",system,agent,500,100)

def test_reinforcement_learning_reacher():
  """
  https://gymnasium.farama.org/environments/mujoco/reacher/
  """
  system=gymnasium_system.GymnasiumSystem("Reacher-v5")
  agent=CustomAgent({"beta":.9},system.state_space(),system.action_space())
  # agent=SB3Agent({"beta":.9},system.state_space(),system.action_space())
  run_test("reacher",system,agent,50,100)

def test_reinforcement_learning_pusher():
  """
  https://gymnasium.farama.org/environments/mujoco/pusher/
  """
  system=gymnasium_system.GymnasiumSystem("Pusher-v5")
  agent=CustomAgent({"beta":.9},system.state_space(),system.action_space())
  run_test("pusher",system,agent,100,100)

def test_reinforcement_learning_ant():
  """
  https://gymnasium.farama.org/environments/mujoco/ant/
  """
  system=gymnasium_system.GymnasiumSystem("Ant-v5")
  agent=CustomAgent({"beta":.98,"update":debug_value("UPDATE",1)==1},system.state_space(),system.action_space())
  run_test("ant",system,agent,1000,100)

def test_reinforcement_learning_humanoid():
  """
  https://gymnasium.farama.org/environments/mujoco/humanoid/
  """
  system=gymnasium_system.GymnasiumSystem("Humanoid-v5")
  agent=CustomAgent({"beta":.98,"database":{"max_size":250000}},system.state_space(),system.action_space())
  run_test("humanoid",system,agent,1000,1000)

def test_reinforcement_learning_double_pendulum():
  system=pendulum_system.PendulumSystem(2,visualize=debug_visualize())
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("double_pendulum",system,agent,1000,100)

def test_reinforcement_learning_pendubot():
  def feedback(model,data):
    # f=0
    # qpos=data.qpos[model.body(1).jntadr[0]:model.body(1).jntadr[0]+7]
    # # print(qpos)
    # f+=qpos[2]
    # return np.array(f)
    print(data.qpos)
    print(data.qvel)
    assert False

  def initialize(model,data,rand):
    print("doing init")
    pass

  xml_path=os.path.join(os.path.dirname(__file__),"../data/pendubot/pendubot.xml")
  system=MujocoSystem(xml_path=xml_path,visualize=debug_visualize(),feedback_function=feedback,initialize_function=initialize)
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("pendubot",system,agent,500,100)

def test_reinforcement_learning_agility_cassie():
  def feedback(model,data):
    f=0
    qpos=data.qpos[model.body(1).jntadr[0]:model.body(1).jntadr[0]+7]
    # print(qpos)
    f+=qpos[2]
    return np.array(f)

  def initialize(model,data,rand):
    print("doing init")
    pass

  xml_path=os.path.join(os.path.dirname(__file__),"../data/agility_cassie/scene.xml")
  system=MujocoSystem(xml_path=xml_path,visualize=debug_visualize(),feedback_function=feedback,initialize_function=initialize)
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("agility_cassie",system,agent,500,100)

def test_reinforcement_learning_jumper():
  def feedback(model,data):
    f=0

    # Have center of mass in the air
    com_z_goal = 0.33
    x_com = data.body('torso').subtree_com
    f += -(x_com[2]-com_z_goal)**2

    # Have foot on ground
    foot_z_goal = 0.02
    x_foot = data.geom('foot_geom').xpos
    # ic( x_foot )
    f += -(x_foot[2]-foot_z_goal)**2

    return np.array(f)

  def initialize(model,data,rand):
    data.qpos[6] = 0.4
    data.qpos[7] = -0.8
    data.qpos[8] = 0.4

  xml_path=os.path.join(os.path.dirname(__file__),"../data/jumper/jumper3d.xml")
  system=MujocoSystem(xml_path=xml_path,visualize=debug_visualize(),feedback_function=feedback,initialize_function=initialize)
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("jumper",system,agent,500,100)

def test_reinforcement_learning_pendulum_triple():
  def feedback(model,data):
    # x_com_goal = np.array([0.    , 0.0657, 0.8952]) # 3 bar
    x_com_goal = np.array([0.        , 0.12383766, 0.74798442]) # 2 bar
    x_com = data.body('link_0').subtree_com
    # ic( x_com )
    f = -np.linalg.norm(x_com-x_com_goal) # - 0.2*np.linalg.norm(data.qvel)
    return np.array(f)

  def initialize(model,data,rand):
    ang = 40*np.pi/180
    # data.qpos = np.hstack([ang, -2*ang, ang])
    data.qpos = np.hstack([ang, -2*ang])

  xml_path=os.path.join(os.path.dirname(__file__),"../data/pendulum_triple/pendulum_triple.xml")
  system=MujocoSystem(xml_path=xml_path,visualize=debug_visualize(),feedback_function=feedback,initialize_function=initialize)
  agent=CustomAgent({"beta":.98},system.state_space(),system.action_space())
  run_test("pendulum_triple",system,agent,2000,100)
