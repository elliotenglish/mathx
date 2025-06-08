import numpy as np
import paramak
import math

# from mathx.geometry.torus import Torus
# from mathx.geometry.cylinder import Cylinder
from mathx.geometry import curvilinear
from mathx.geometry.fourier import FourierND
from mathx.geometry import basis
from mathx.core import log
from mathx.core.jax_utilities import Generator
from .toroidal_plasma import ToroidalPlasma
# from .magnet import Magnet

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import functools

@dataclass
class ReactorParameters:
  wall_thickness: float
  magnets_conformal_num: int
  magnets_conformal_width: float
  magnets_ring_num: int
  magnets_ring_radius: float
  magnets_ring_width: float
  magnets_cylinder_num: int
  magnets_cylinder_radius: float
  magnets_cylinder_length: float
  magnets_cylinder_thickness: float
  magnets_cylinder_phase: float
  ports_num: int
  ports_length: float
  ports_radius: float
  ports_thickness: float
  supports_num: int
  supports_width: float
  supports_ground_level: float

def PlasmaSurface(plasma,surface):
  log.info(f"Plasma {surface=}")
  return curvilinear.Curvilinear(
    lambda u: plasma.get_surface(
      jnp.array([u[0]*surface,u[1]*2*jnp.pi,u[2]*2*jnp.pi]))[0],
    closed=(False,True,True),
    degenerate=(((False,True),None),None,None))

def PlasmaChamber(structure_fn,thickness):
  log.info(f"PlasmaChamber {thickness=}")
  return curvilinear.Curvilinear(
    lambda u: structure_fn(jnp.array([u[0],u[1],u[2]*thickness]))[0],
    closed=(True,True,False),
    min_segments=(4,4,1))

def ConformalMagnet(structure_fn,width):
  # log.info(f"RingMagnet {width=}")

  def pos_fn(u):
    x,b,n=structure_fn(jnp.array([0,u[0],0]))
    dphi=b[:,0]/jnp.linalg.norm(b[:,0])
    xp=x+dphi*width*(u[1]-.5)+n*width*u[2]

    return xp

  return curvilinear.Curvilinear(
    pos_fn,
    closed=(True,False,False),
    min_segments=(4,1,1))

def RingMagnet(x,basis,radius,width):
  def pos_fn(u):
    r=radius+u[2]*width
    return (x+
            (u[0]-.5)*width*basis[:,0]+
            jnp.cos(2*jnp.pi*u[1])*r*basis[:,1]+
            jnp.sin(2*jnp.pi*u[1])*r*basis[:,2])
    
  return curvilinear.Curvilinear(
    pos_fn,
    closed=(False,True,False),
    min_segments=(1,4,1))

def CylinderMagnet(x,basis,radius,length,thickness):
  def pos_fn(u):
    r=radius+u[2]*thickness
    return (x+
        (u[0]-.5)*length*basis[:,0]+
        jnp.cos(2*jnp.pi*u[1])*r*basis[:,1]+
        jnp.sin(2*jnp.pi*u[1])*r*basis[:,2])
    
  return curvilinear.Curvilinear(
    pos_fn,
    closed=(False,True,False),
    min_segments=(1,4,1))

def Port(structure_fn,uc,length,radius,thickness):
  log.info(f"Port {uc=} {length=} {radius=} {thickness=}")
  xc,bc,nc=structure_fn(uc)
  
  def pos_fn(u):
    r=radius+thickness*u[2]
    ua=uc+jnp.array([jnp.cos(2*jnp.pi*u[1])*r/jnp.linalg.norm(bc[:,0]),
                     jnp.sin(2*jnp.pi*u[1])*r/jnp.linalg.norm(bc[:,1]),
                     0])
    xa=structure_fn(ua)[0]
    # jax.debug.print("uc={uc} ua={ua} xa={xa}",uc=uc,ua=ua,xa=xa)
    return xa+u[0]*length*nc
  
  return curvilinear.Curvilinear(pos_fn,closed=(False,True,False))

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

class Reactor:
  # @functools.partial(jax.jit,static_argnums=(0,))
  # def plasma_fn(self,u):
  #   def pure_callback(u):
  #     us=u[None,:] if len(u.shape)==1 else u
  #     xs,bs=equilibrium.get_xyz_basis(self.plasma_equilibrium,us)
  #     xs,bs=(xs[0],bs[0]) if len(u.shape)==1 else (xs,bs)
  #     return xs,bs

  #   x,b=jax.pure_callback(pure_callback,
  #                         (jax.ShapeDtypeStruct((3,),u.dtype),
  #                          jax.ShapeDtypeStruct((3,3),u.dtype)),
  #                         u,
  #                         vmap_method="expand_dims")
  #   return x,b

  # @functools.partial(jax.jit,static_argnums=(0,))
  def structure_fn_plasma(self,u):
    # Handle singularity
    epsilon=1e-5
    u=jnp.array([u[0],u[1],jnp.maximum(epsilon,u[2])])

    up=u[::-1]*jnp.array([1,2*jnp.pi,2*jnp.pi])+jnp.array([1,0,0])

    x,bp=self.plasma.get_surface(up)
    b=bp[:,::-1]*jnp.array([1./(2*jnp.pi),1./(2*jnp.pi),1])[None,...]

    n=jnp.cross(b[:,1],b[:,0])
    n=n/jnp.linalg.norm(n)
    x=x+u[2]*n

    # jax.debug.print("u={u} up={up} n={n} x={x}",u=u,up=up,n=n,x=x)
    # jax.debug.print("b={b}",b=b)

    return x,b,n

  def get_basis_bounds(self,b,xs):
    db=xs@b
    low=np.min(db,axis=0)
    high=np.max(db,axis=0)
    return low,high

  def get_structure_bounds(self, phi, length):
    # import pdb
    # pdb.set_trace()
    num=8
    thetas=jnp.linspace(0,1,num,endpoint=False)
    us=jnp.concatenate([jnp.array([[phi]]*num),
                        thetas[...,None],
                        jnp.array([[0]]*num)],
                       axis=1)
    # us=jnp.array([[0,0,0]])
    xs,bs,ns=self.structure_fn_batch(us)
    # log.info(f"{us=}")
    # log.info(f"{xs=}")
    # log.info(f"{bs=}")
    b=np.sum(bs,axis=0)
    b=basis.orthogonalize(b.T).T
    # log.info(f"{b=}")
    low,high=self.get_basis_bounds(b,xs)
    # log.info(f"structure {phi=} {low=} {high=}")
    x=b@(low+high)/2
    radius=np.linalg.norm(high[1:3]-low[1:3])/2
    return x,b,radius

    # dxdphi=jnp.max(jnp.linalg.norm(bs[:,:,0],axis=1))
    # dphi=.5*length/dxdphi
    # us=jnp.concatenate([[[phi]],thetas[None,...],[[0]]])

  def __init__(self, plasma, params: ReactorParameters):
    self.rand=np.random.default_rng(54323)

    self.plasma=plasma

    self.structure_fn=self.structure_fn_plasma
    self.structure_fn_batch=jax.jit(jax.vmap(self.structure_fn,in_axes=(0),out_axes=(0,0,0)))

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
    self.plasma_surface=PlasmaSurface(self.plasma,1)

    ###################################
    # Plasma chamber
    log.info("creating plasma chamber")
    # self.plasma_chamber=Torus(params.plasma_chamber)
    self.plasma_chamber=PlasmaChamber(self.structure_fn,params.wall_thickness)

    #Offset structure function by chamber wall thickness
    log.info("defining structure function")
    self.structure_fn_offset=lambda u:self.structure_fn(u+jnp.array([0,0,params.wall_thickness]))

    # print(self.structure_fn(jnp.array([0.,0,0])))
    # print(self.structure_fn(jnp.array([0,.25,0])))
    # print(self.structure_fn(jnp.array([0,.5,0])))
    # print(self.structure_fn(jnp.array([0,.75,0])))
    # import pdb
    # pdb.set_trace()

    ###################################
    # Magnets
    self.magnets=[]

    ###############
    log.info("generating conformal magnets")
    magnets_conformal_phi=jnp.linspace(0,1,params.magnets_conformal_num,endpoint=False)
    n_mode=2
    modes=[(m*plasma.nfp,n)
           for m in range(-n_mode,n_mode+1)
           for n in range(-n_mode,n_mode+1)]
    coefficients=Generator(423).uniform(size=[len(modes)],low=-1,high=1)*.005
    pert=FourierND(modes=modes,coefficients=coefficients)

    # Use a function so that phi is copied into the function closure. A lambda doesn't do this.
    def GenerateConformalMagnet(phi):
      return ConformalMagnet(
        lambda u:self.structure_fn_offset(u+jnp.array([phi+pert(jnp.array([phi,u[1]])),0,0])),
        # lambda u:self.structure_fn_offset(u+jnp.array([phi,0,0])),
        params.magnets_conformal_width)

    # log.info(f"{magnet_phi=}")
    self.magnets+=[
      GenerateConformalMagnet(phi)
      for phi in magnets_conformal_phi
    ]

    ###############
    log.info("generating ring magnets")
    def GenerateRingMagnet(phi):
      x,b,r=self.get_structure_bounds(float(phi),params.magnets_ring_width)
      return RingMagnet(x,b,r+params.magnets_ring_radius,params.magnets_ring_width)

    magnets_ring_phi=jnp.linspace(0,1,params.magnets_ring_num,endpoint=False)
    self.magnets+=[
      GenerateRingMagnet(phi)
      for phi in magnets_ring_phi]
    
    ###############
    log.info("generating cylinder magnets")
    def GenerateCylinderMagnet(phi):
      x,b,r=self.get_structure_bounds(float(phi),params.magnets_cylinder_length)
      # log.info(f"{phi=} {x=} {b=} {r=}")
      # x,b=self.plasma.get_surface(jnp.array([1e-5,0,2*jnp.pi*(phi+params.magnets_cylinder_phase)]))
      # b=b[:,::-1]
      # b=basis.orthogonalize(b.T).T
      return CylinderMagnet(x,b,
                            r+params.magnets_cylinder_radius,
                            params.magnets_cylinder_length,
                            params.magnets_cylinder_thickness)

    magnets_cylinder_phi=jnp.linspace(0,1,params.magnets_cylinder_num,endpoint=False)
    self.magnets+=[
      GenerateCylinderMagnet(phi)
      for phi in magnets_cylinder_phi]
    
    ###################################
    # Ports
    def GeneratePort(u):
      # x,b,n=self.structure_fn_offset(u)
      return Port(self.structure_fn_offset,
                  u,
                  params.ports_length,
                  params.ports_radius,
                  params.ports_thickness)

    self.ports=[]
    while len(self.ports)<params.ports_num:
      u=np.concatenate([self.rand.uniform(size=2,low=0,high=1),[0]])
      self.ports.append(GeneratePort(u))

    ###################################
    # Supports
    log.info("generating supports")
    self.supports=[]
    # while len(self.supports)<params.supports_num:
    for phi in jnp.linspace(0,1,params.supports_num,endpoint=False):
      # u=np.concatenate([rand.uniform(low=0,high=1,size=[2]),[0]])
      # self.rand.uniform(low=0,high=1,size=[1]),[1]
      for theta in jnp.linspace(0,1,16):
        u=jnp.array([phi,theta,0])
        x,b,n=self.structure_fn_offset(u)
        print(u,x,n)
        if n[2]<-.8:
          log.info(f"support {u=} {x=} {n=}")
          self.supports.append(Support(self.structure_fn_offset,u,params.supports_ground_level,params.supports_width))
          break

    ###################################
    # Components
    self.components=(
      [
        self.plasma_surface,
        self.plasma_chamber
      ] +
      self.magnets +
      self.ports +
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
