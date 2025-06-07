import numpy as np
import jax.numpy as jnp
# import math
from dataclasses import dataclass

from .curvilinear import Curvilinear

def toroid_to_xyz(major_radius,minor_radius_inner,minor_radius_outer,rho,theta,phi):
  """
  params:
    R: major radius, the distance from the origin to the center of the toroid ring
    r_inner: minor radius, the distance from the center of the toroid ring to the inner surface
    r_outer: minor radius, the distance form the center of the toroid ring to the outer surface
    rho: fraction of minor radius
    theta: toroidal angle
    phi: poloidal angle
  """

  r=minor_radius_inner+rho*(minor_radius_outer-minor_radius_inner)
  R=major_radius+r*jnp.cos(-theta)
  z=r*jnp.sin(-theta)
  x=R*jnp.cos(phi)
  y=R*jnp.sin(phi)
  return x,y,z

def xyz_to_toroid(major_radius,minor_radius_inner,minor_radius_outer,x,y,z):
  phi=jnp.arctan2(y,x)
  R=jnp.sqrt(x**2+y**2)
  theta=-jnp.arctan2(z,R-major_radius)
  r=jnp.sqrt(z**2+(R-major_radius)**2)
  rho=(r-minor_radius_inner)/(minor_radius_outer-minor_radius_inner)
  return rho,theta,phi

@dataclass
class Parameters:
  major_radius: float
  minor_radius_inner: float
  minor_radius_outer: float

def Torus(params: Parameters = None, **kwargs):
  if params is None:
    params=Parameters(**kwargs)
  return Curvilinear(lambda u:jnp.array(
    toroid_to_xyz(params.major_radius,
                  params.minor_radius_inner,
                  params.minor_radius_outer,
                  u[2],
                  u[1]*2*jnp.pi,
                  u[0]*2*jnp.pi)),
      closed=(True,True,False),
      min_segments=(4,4,1),
      degenerate=(None,None,((False,True if params.minor_radius_inner==0 else False),None)))

# def compute_density(self,num):
#   toroidal_circumference=math.pi*2*self.params.major_radius
#   poloidal_circumference=math.pi*2*(self.params.minor_radius_inner+self.params.minor_radius_outer)/2
#   num_toroidal=max(4,num)
#   #The 2 multiplier here is a fudge factor. We really need to look at curvature/edge angles
#   num_poloidal=max(4,2*math.ceil(poloidal_circumference/toroidal_circumference*num))
#   num_radial=math.ceil((self.params.minor_radius_outer-self.params.minor_radius_inner)/toroidal_circumference*num)
#   print(f"nt={num_toroidal} np={num_poloidal} nr={num_radial}")
#   return num_toroidal,num_poloidal,num_radial
