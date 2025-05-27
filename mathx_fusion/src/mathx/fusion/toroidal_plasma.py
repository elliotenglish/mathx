import abc
from . import torus
import jax.numpy as jnp

def ToroidalPlasma(abc.ABC):
  @abc.abstractmethod
  def get_surface(self,u):
    """
    params:
      u: Reference space position (toroidal [0-1], poloidal [0-1], radial [0-1])
    returns:
      x: Physical space position
      dx/dy: Basis
    """
    pass

  @abc.abstractmethod
  def get_B(self,u):
    pass

  @abc.abstractmethod
  def get_u(self,x):
    pass

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
