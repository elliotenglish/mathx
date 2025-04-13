from mathx.learning.mujoco_system import MujocoSystem
from test_config import *

import os

def test_mujoco_system():
  # xml_path=os.path.join(os.path.dirname(__file__),"../data/agility_cassie/cassie.xml")
  xml_path=os.path.join(os.path.dirname(__file__),"../data/agility_cassie/scene.xml")
  sys=MujocoSystem(xml_path=xml_path,visualize=debug_visualize())
  while True:
    #print(sys.state())
    sys.transition(None)
  
if __name__=="__main__":
  test_mujoco_system()
