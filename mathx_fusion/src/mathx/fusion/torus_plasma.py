from mathx.geometry import torus
from mathx.geometry import curvilinear
from .toroidal_plasma import ToroidalPlasma
import jax.numpy as jnp

class TorusPlasma(ToroidalPlasma):
  def __init__(self,major_radius,minor_radius):
    self.major_radius=major_radius
    self.minor_radius=minor_radius

  def get_surface(self,rtz):
    def xyz_fn(rtp):
      return jnp.array(torus.toroid_to_xyz(self.major_radius,0,self.minor_radius,rtp[0],rtp[1],rtp[2]))

    basis_fn=curvilinear.contravariant_basis_transform(xyz_fn)
    xyz=xyz_fn(rtz)
    basis=basis_fn(rtz)

    return xyz,basis

  def get_B(self,rtz):
    return jnp.array([0,jnp.pi*.05,.05])

  def get_u(self,x):
    rtz=torus.xyz_to_toroid(self.major_radius,0,self.minor_radius,*x)
    return jnp.array(rtz)
