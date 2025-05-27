from . import torus
from .toroidal_plasma import ToroidalPlasma
import jax.numpy as jnp

def TorusPlasma(ToroidalPlasma):
  def __init__(self,major_radius,minor_radius):
    self.major_radius=major_radius,
    self.minor_radius=minor_radius

  def get_surface(self,u):
    rtp=[u[2],u[1]*2*jnp.pi,u[0]*2*jnp.pi]
    xyz=torus.toroid_to_xyz(self.major_radius,0,self.minor_radius,
                            *rtp)
    return jnp.array(xyz)

  def get_B(self,u):
    return jnp.array(1,0,0)
  
  def get_u(self,x):
    rtp=torus.xyz_to_toroid(self.major_radius,0,self.minor_radius,*x)
    return jnp.array([rtp[0]/(2*jnp.pi),rtp[1]/(2*jnp.pi),rtp[2]])
