import numpy as np
import math
from dataclasses import dataclass

from .curvilinear import Curvilinear

def cylinder_to_xyz(rho,phi,z):
  """
  params:
    rho: distance from axis
    phi: angle
    z: vertical position
  """

  x=rho*math.cos(phi)
  y=rho*math.sin(phi)

  return x,y,z

@dataclass
class Parameters:
  radius: float
  thickness: float
  length: float

def Cylinder(params: Parameters = None, **kwargs):
  if params is None:
    params=Parameters(**kwargs)
  return Curvilinear(lambda u:np.array(
    cylinder_to_xyz(params.radius+u[2]*params.thickness,
                    u[1]*2*math.pi,
                    u[0]*params.length)),
    closed=(False,True,False),
    degenerate=(None,None,((False,True if params.radius==0 else False),None)))

#   def generate_density(self,num):
#     circumference=math.pi*2*(self.params.radius+self.params.thickness/2)
#     num_azimuth=max(4,num)
#     num_length=math.ceil(self.params.length/circumference*num)
#     num_radial=math.ceil(self.params.thickness/circumference*num)
#     # num_length=1#
#     # num_radial=1#
#     print(f"na={num_azimuth} nl={num_length} nr={num_radial}")
#     return num_length,num_azimuth,num_radial
