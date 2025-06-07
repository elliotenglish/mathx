import abc
import jax.numpy as jnp

class ToroidalPlasma(abc.ABC):
  @abc.abstractmethod
  def get_surface(self,rtz):
    """
    params:
      rtz: Reference space position (radial [0,1], poloidal [0,2*\pi], toroidal [0,2*\pi])
    returns:
      x: Physical space position
      dx/dy: Basis
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_B(self,rtz):
    raise NotImplementedError

  @abc.abstractmethod
  def get_u(self,x):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def nfp(self):
    raise NotImplementedError
