import numpy as np
import math
from dataclasses import dataclass

from . import curvilinear

def toroid_to_xyz(R,r,theta,phi):
  """
  params:
    R: major radius, the distance from the origin to the center of the toroid ring
    r: minor radius, the distance from the center of the toroid ring to the toroid surface
    theta: toroidal angle
    phi: poloidal angle
  """

  rl=R+r*math.cos(theta)
  z=r*math.sin(theta)
  x=rl*math.cos(phi)
  y=rl*math.sin(phi)

  return (x,y,z)

class Torus:
  @dataclass
  class Parameters:
    major_radius: float
    minor_radius: float
    thickness: float

  def __init__(self,
               params: Parameters = None,
               **kwargs):
    if params is not None:
      self.params=params
    else:
      self.params=Torus.Parameters(**kwargs)
      
  def compute_density(self,num):
    toroidal_circumference=math.pi*2*self.params.major_radius
    poloidal_circumference=math.pi*2*(self.params.minor_radius+self.params.thickness/2)
    num_toroidal=max(4,num)
    num_poloidal=max(4,math.ceil(poloidal_circumference/toroidal_circumference*num))
    num_radial=math.ceil(self.params.thickness/toroidal_circumference*num)
    print(f"nt={num_toroidal} np={num_poloidal} nr={num_radial}")
    return num_toroidal,num_poloidal,num_radial

  def tesselate_volume(self,density):
    segments=self.compute_density(density)

    return curvilinear.generate_mesh_volume(
      lambda u,v,w:toroid_to_xyz(self.params.major_radius,
                                 self.params.minor_radius+w*self.params.thickness,
                                 v*2*math.pi,
                                 u*2*math.pi),
      segments,
      closed=(True,True,False),
      degenerate=(None,None,((False,True if self.params.minor_radius==0 else False),None)))
  
  def tesselate_surface(self,density):
    num_toroidal,num_poloidal,num_radial=self.compute_density(density)
