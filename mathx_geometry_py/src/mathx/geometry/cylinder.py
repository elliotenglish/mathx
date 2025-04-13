import numpy as np
import math
from dataclasses import dataclass

from . import curvilinear

def cylinder_to_xyz(rho,phi,z):
  """
  params:
    rho: distance from axis
    phi: angle
    z: vertical position
  """

  x=rho*math.cos(phi)
  y=rho*math.sin(phi)

  return (x,y,z)

class Cylinder:
  @dataclass
  class Parameters:
    radius: float
    thickness: float
    length: float

  def __init__(self,
               params: Parameters = None,
               **kwargs):
    if params is not None:
      self.params=params
    else:
      self.params=Cylinder.Parameters(**kwargs)

  def generate_density(self,num):
    circumference=math.pi*2*(self.params.radius+self.params.thickness/2)
    num_azimuth=max(4,num)
    num_length=math.ceil(self.params.length/circumference*num)
    num_radial=math.ceil(self.params.thickness/circumference*num)
    # num_length=1#
    # num_radial=1#
    print(f"na={num_azimuth} nl={num_length} nr={num_radial}")
    return num_length,num_azimuth,num_radial

  def tesselate_volume(self,density):
      # Attempt to produce square elements
    segments=self.generate_density(density)

    return curvilinear.generate_mesh_volume(
      lambda u,v,w:cylinder_to_xyz(self.params.radius+w*self.params.thickness,v*2*math.pi,u*self.params.length),
      segments,
      closed=(False,True,False),
      degenerate=(None,None,((False,True if self.params.radius==0 else False),None)))
