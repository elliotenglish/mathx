import numpy as np
import paramak
import math

from mathx.geometry.torus import Torus
from mathx.geometry.cylinder import Cylinder
from mathx.geometry import curvilinear
from mathx.fusion import equilibrium
from mathx.core import log
# from .magnet import Magnet

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import functools

def RingMagnet(structure_fn,phi,width,height):
  def pos_fn(u):
    x,b=structure_fn(jnp.array([phi,u[0],0]))
    dphi=b[:,0]/jnp.linalg.norm(b[:,0])
    dtheta=b[:,1]/jnp.linalg.norm(b[:,1])
    # dtheta=b[:,1]-(b[:,1]@dphi.T)*dphi
    # dtheta=dtheta/jnp.linalg.norm(dtheta)
    norm=jnp.cross(dphi,dtheta)
    norm=norm/jnp.linalg.norm(norm)
    return x+dphi*width*(u[1]-.5)+norm*height*u[2]

  return curvilinear.Curvilinear(
    pos_fn,
    closed=(True,False,False),
    min_segments=(4,1,1))

def PlasmaChamber(structure_fn,thickness):
  return curvilinear.Curvilinear(
      lambda u: structure_fn(jnp.array([u[0],u[1],u[2]*thickness]))[0],
      closed=(True,True,False),
      min_segments=(4,4,1))

@dataclass
class ReactorParameters:
  pass
  # primary_chamber: Torus.Parameters
  # magnets: list[Magnet.Parameters]
  # heating_ports: list[Port.Parameters]
  # diagnostic_ports: list[Port.Parameters]
  # diverter_ports: list[Port.Parameters]


class Reactor:
  def surface_fn(self,u):
    return self.primary_chamber.pos_fn(jnp.concatenate([u,np.array([1])]))

  @functools.partial(jax.jit,static_argnums=(0,))
  def structure_fn(self,u):
    norm_fn=curvilinear.surface_normal_transform(self.surface_fn)
    x=self.surface_fn(u[:2])
    n=norm_fn(u[:2])
    return x+n*(u[2]+jnp.sin(u[0]*8*jnp.pi)*.1)

  @functools.partial(jax.jit,static_argnums=(0,))
  def structure_fn_plasma(self,u):
    # log.info(f"{u=} {u.shape}")
    def structure_fn_plasma_callback(u):
      us=u[None,:] if len(u.shape)==1 else u
      # log.info(f"{us.shape=}")
      # log.info(f"{us=}")
      # log.info(f"computing vertices num={us.shape[0]}")
      xs,bs=equilibrium.get_xyz_basis(self.plasma_equilibrium,us)
      # log.info(f"computed")
      # log.info(f"{xs=}")
      # log.info(f"{bs=}")
      # log.info(f"{xs.shape=} {bs.shape=}")
      xs,bs=(xs[0],bs[0]) if len(u.shape)==1 else (xs,bs)
      return xs,bs

    u=jnp.array([u[0],u[1],1])

    x,b=jax.pure_callback(structure_fn_plasma_callback,
                          (jax.ShapeDtypeStruct((3,),u.dtype),
                           jax.ShapeDtypeStruct((3,3),u.dtype)),
                          u,
                          vmap_method="expand_dims")
    n=jnp.cross(b[:,0],b[:,1])
    n=n/jnp.linalg.norm(n)
    x=x+u[2]*n

    return x,b

  def __init__(self, params: ReactorParameters):
    self.plasma_equilibrium=equilibrium.get_test_equilibrium()

    # us0=jnp.array([[0,0,0]],dtype=jnp.float64)
    # us1=jnp.array([[0,0,0],[.5,0,0]],dtype=jnp.float64)
    # log.info(f"direct {equilibrium.get_xyz(self.plasma_equilibrium,us0)}")
    # log.info(f"direct batched {equilibrium.get_xyz(self.plasma_equilibrium,us1)}")

    # log.info(f"indirect {self.structure_fn_plasma(us0[0])}")
    # log.info(f"indirect batched {jax.jax.vmap(self.structure_fn_plasma,in_axes=(0),out_axes=(0))(us1)}")

    # log.info(f"jit {jax.jit(self.structure_fn_plasma)(us0[0])}")
    # log.info(f"jit batched {jax.jit(jax.vmap(self.structure_fn_plasma,in_axes=(0),out_axes=(0)))(us1)}")

    # import sys
    # sys.exit(0)

    log.info("creating components")

    # self.plasma_chamber=Torus(params.plasma_chamber)
    self.wall_thickness=.2
    self.plasma_chamber=PlasmaChamber(self.structure_fn_plasma,self.wall_thickness)

    magnet_phi=jnp.linspace(0,1,32,endpoint=False)
    log.info(f"{magnet_phi=}")
    self.magnets=[
      RingMagnet(self.structure_fn_plasma,phi,.1,.1)
      for phi in magnet_phi
    ]

    self.density=128

  def generate(self):
    log.info("meshing")

    return (
      [self.plasma_chamber.tesselate_surface(self.density)] +
      [m.tesselate_surface(self.density) for m in self.magnets]
    )

    #1 Define plasma chamber using surface_fn
    #2 Define magnets as grid on offset surface_fn
    #3 Define ports using surface_fn
    #4 Define support stucture by sampling extrema of surface function and then extruding to ground plane
    #4 Define exterior mechanisms...
