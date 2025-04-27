import numpy as np
import math
from dataclasses import dataclass

from . import curvilinear

def toroid_to_xyz(R,r_inner,r_outer,rho,theta,phi):
  """
  params:
    R: major radius, the distance from the origin to the center of the toroid ring
    r_inner: minor radius, the distance from the center of the toroid ring to the inner surface
    r_outer: minor radius, the distance form the center of the toroid ring to the outer surface
    rho: fraction of minor radius
    theta: toroidal angle
    phi: poloidal angle
  """

  r_=r_inner+rho*(r_outer-r_inner)
  rl=R+r_*math.cos(theta)
  z=r_*math.sin(theta)
  x=rl*math.cos(phi)
  y=rl*math.sin(phi)

  return (x,y,z)

class Torus:
  @dataclass
  class Parameters:
    major_radius: float
    minor_radius_inner: float
    minor_radius_outer: float

  def __init__(self,
               params: Parameters = None,
               **kwargs):
    if params is not None:
      self.params=params
    else:
      self.params=Torus.Parameters(**kwargs)
      
  def compute_density(self,num):
    toroidal_circumference=math.pi*2*self.params.major_radius
    poloidal_circumference=math.pi*2*(self.params.minor_radius_inner+self.params.minor_radius_outer)/2
    num_toroidal=max(4,num)
    #The 4 multiplier here is a fudge factor. We really need to look at curvature/edge angles
    num_poloidal=max(4,4*math.ceil(poloidal_circumference/toroidal_circumference*num))
    num_radial=math.ceil((self.params.minor_radius_outer-self.params.minor_radius_inner)/toroidal_circumference*num)
    print(f"nt={num_toroidal} np={num_poloidal} nr={num_radial}")
    return num_toroidal,num_poloidal,num_radial

  def tesselate_volume(self,density):
    segments=self.compute_density(density)

    return curvilinear.generate_mesh_volume(
      lambda u,v,w:toroid_to_xyz(self.params.major_radius,
                                 self.params.minor_radius_inner,
                                 self.params.minor_radius_outer,
                                 w,
                                 v*2*math.pi,
                                 u*2*math.pi),
      segments,
      closed=(True,True,False),
      degenerate=(None,None,((False,True if self.params.minor_radius_inner==0 else False),None)))
  
  def tesselate_surface(self,density):
    num_toroidal,num_poloidal,num_radial=self.compute_density(density)
