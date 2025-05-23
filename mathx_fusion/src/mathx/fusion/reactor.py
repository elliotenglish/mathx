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

def Plasma(structure_fn,surface):
  log.info(f"Plasma {surface=}")
  return curvilinear.Curvilinear(
    lambda u: structure_fn(jnp.array([u[0],u[1],u[2]*surface]))[0],
    closed=(True,True,False),
    degenerate=(None,None,((False,True),None)))

def PlasmaChamber(structure_fn,thickness):
  log.info(f"PlasmaChamber {thickness=}")
  return curvilinear.Curvilinear(
    lambda u: structure_fn(jnp.array([u[0],u[1],u[2]*thickness]))[0],
    closed=(True,True,False),
    min_segments=(4,4,1))

def RingMagnet(structure_fn,phi,width,height):
  log.info(f"RingMagnet {phi=} {width=} {height=}")
  def pos_fn(u):
    x,b,n=structure_fn(jnp.array([phi,u[0],0]))
    dphi=b[:,0]/jnp.linalg.norm(b[:,0])
    return x+dphi*width*(u[1]-.5)+n*height*u[2]

  return curvilinear.Curvilinear(
    pos_fn,
    closed=(True,False,False),
    min_segments=(4,1,1))

def Support(structure_fn,uc,ground_level,width):
  log.info(f"Support {uc=} {ground_level=} {width=}")
  xc,bc,nc=structure_fn(uc)

  def pos_fn(u):
    ua=uc+jnp.array([(u[0]-.5)*width/jnp.linalg.norm(bc[:,0]),
                     (u[1]-.5)*width/jnp.linalg.norm(bc[:,1]),
                     0])
    xa=structure_fn(ua)[0]
    return jnp.array([xa[0],xa[1],ground_level*(1-u[2])+u[2]*xa[2]])

  return curvilinear.Curvilinear(pos_fn)

@dataclass
class ReactorParameters:
  wall_thickness: float = .2
  magnet_width: float =.2
  magnet_height: float = .2
  num_supports: int = 24
  # primary_chamber: Torus.Parameters
  # magnets: list[Magnet.Parameters]
  # heating_ports: list[Port.Parameters]
  # diagnostic_ports: list[Port.Parameters]
  # diverter_ports: list[Port.Parameters]

class Reactor:
  # @functools.partial(jax.jit,static_argnums=(0,))
  def structure_fn_torus(self,u):
    # norm_fn=curvilinear.surface_normal_transform(self.surface_fn)
    # x=self.surface_fn(u[:2])
    # n=norm_fn(u[:2])
    # return x+n*(u[2]+jnp.sin(u[0]*8*jnp.pi)*.1)
    assert False

  # @functools.partial(jax.jit,static_argnums=(0,))
  def plasma_fn(self,u):
    def pure_callback(u):
      us=u[None,:] if len(u.shape)==1 else u
      xs,bs=equilibrium.get_xyz_basis(self.plasma_equilibrium,us)
      xs,bs=(xs[0],bs[0]) if len(u.shape)==1 else (xs,bs)
      return xs,bs

    x,b=jax.pure_callback(pure_callback,
                          (jax.ShapeDtypeStruct((3,),u.dtype),
                           jax.ShapeDtypeStruct((3,3),u.dtype)),
                          u,
                          vmap_method="expand_dims")
    return x,b

  # @functools.partial(jax.jit,static_argnums=(0,))
  def structure_fn_plasma(self,u):
    up=jnp.array([u[0],u[1],1])

    # import pdb
    # pdb.set_trace()

    x,b=self.plasma_fn(up)
    n=-jnp.cross(b[:,0],b[:,1])
    n=n/jnp.linalg.norm(n)
    x=x+u[2]*n

    return x,b,n

  def __init__(self, params: ReactorParameters):
    params.wall_thickness=0.2
    self.plasma_equilibrium=equilibrium.get_test_equilibrium()

    self.structure_fn=self.structure_fn_plasma

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

    ###################################
    # Plasma
    self.plasma=Plasma(self.plasma_fn,1)

    ###################################
    # Plasma chamber
    # self.plasma_chamber=Torus(params.plasma_chamber)
    self.plasma_chamber=PlasmaChamber(self.structure_fn,params.wall_thickness)

    self.structure_fn_offset=lambda u:self.structure_fn(u+jnp.array([0,0,params.wall_thickness]))

    # print(self.structure_fn(jnp.array([0.,0,0])))
    # print(self.structure_fn(jnp.array([0,.25,0])))
    # print(self.structure_fn(jnp.array([0,.5,0])))
    # print(self.structure_fn(jnp.array([0,.75,0])))
    # import pdb
    # pdb.set_trace()

    ###################################
    # Magnets
    num_magnets=32
    magnet_phi=jnp.linspace(0,1,num_magnets,endpoint=False)
    # log.info(f"{magnet_phi=}")
    self.magnets=[
      RingMagnet(self.structure_fn_offset,phi,.2,.2)
      for phi in magnet_phi
    ]

    ###################################
    # Supports
    rand=np.random.default_rng(54323)

    self.supports=[]
    # while len(self.supports)<params.num_supports:
    for phi in np.linspace(0,1,params.num_supports,endpoint=False):
      # u=np.concatenate([rand.uniform(low=0,high=1,size=[2]),[0]])
      while True:
        u=np.concatenate([[phi],rand.uniform(low=0,high=1,size=[1]),[0]])
        x,b,n=self.structure_fn_offset(u)
        # print(u,x,n)
        if n[2]<-.9:
          # log.info(f"support {u=} {x=} {n=}")
          self.supports.append(Support(self.structure_fn_offset,u,-3,.01))
          break

    ###################################
    # Components
    self.components=(
      [
        self.plasma,
        self.plasma_chamber
      ] +
      self.magnets +
      self.supports
    )

    ###################################
    # Meshing
    self.density=64

  def generate(self):
    log.info("meshing")

    return (
      [c.tesselate_surface(self.density) for c in self.components]
    )

    #1 Define plasma chamber using surface_fn
    #2 Define magnets as grid on offset surface_fn
    #3 Define ports using surface_fn
    #4 Define support stucture by sampling extrema of surface function and then extruding to ground plane
    #4 Define exterior mechanisms...
