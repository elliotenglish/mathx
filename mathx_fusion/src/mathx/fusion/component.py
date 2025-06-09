from mathx.geometry.curvilinear import Curvilinear
# from mathx.geometry.mesh import Mesh
from dataclasses import dataclass

@dataclass
class Component:
  object: Curvilinear
  # mesh: Mesh = None
  material: str
