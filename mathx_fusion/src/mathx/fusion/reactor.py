import numpy as np
import paramak
import math

from mathx.geometry.torus import Torus
from mathx.geometry.cylinder import Cylinder
from mathx.geometry import curvilinear
from mathx.fusion import equilibrium
# from .magnet import Magnet

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import functools

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

  def structure_fn_plasma(self,u):
    def structure_fn_plasma_callback(u):
      # print(u)
      us=u[None,:] if len(u.shape)==1 else u
      xs=return equilibrium.get_xyz(self.plasma_equilibrium,us)
      return xs[0] if len(u.shape)==1 else xs
    return jax.pure_callback(structure_fn_plasma_callback,
                             jax.ShapeDtypeStruct(u.shape,u.dtype),
                             u)

  def __init__(self, params: ReactorParameters):
    self.plasma_equilibrium=equilibrium.get_test_equilibrium()

    us0=jnp.array([[0,0,0]],dtype=jnp.float64)
    us1=jnp.array([[0,0,0],[.5,0,0]],dtype=jnp.float64)
    print("direct",equilibrium.get_xyz(self.plasma_equilibrium,us0))
    print("direct batched",equilibrium.get_xyz(self.plasma_equilibrium,us1))

    print("indirect",self.structure_fn_plasma(us0[0]))
    print("indirect batched",jax.jax.vmap(self.structure_fn_plasma,in_axes=(0),out_axes=(0))(us1))
    
    print("jit",jax.jit(self.structure_fn_plasma)(us0[0]))
    print("jit batched",jax.jit(jax.vmap(self.structure_fn_plasma,in_axes=(0),out_axes=(0)))(us1))

    import sys
    sys.exit(0)

    # self.plasma_chamber=Torus(params.plasma_chamber)
    self.primary_chamber=Torus(major_radius=1,minor_radius_inner=0,minor_radius_outer=0.2)
    self.wall_thickness=.05
    self.wall=curvilinear.Curvilinear(
      lambda u: self.structure_fn_plasma(jnp.array([u[0],u[1],u[2]*self.wall_thickness])),
      closed=(True,True,False),
      min_segments=(4,4,1))
    self.density=32

  def generate(self):
    return (
        [self.wall.tesselate_surface(self.density)]
      # + [m.tesselate_surface(self.density) for m in self.magnets]
    )

    #1 Define plasma chamber using surface_fn
    #2 Define magnets as grid on offset surface_fn
    #3 Define ports using surface_fn
    #4 Define support stucture by sampling extrema of surface function and then extruding to ground plane
    #4 Define exterior mechanisms...
