import jax
import desc
for d in jax.devices():
  if d.platform=="gpu":
    desc.set_device("gpu")

# from desc.equilibrium import Equilibrium
# from desc.geometry import FourierRZToroidalSurface
# from desc.profiles import PowerSeriesProfile


# from desc.continuation import solve_continuation_automatic
# from desc.grid import LinearGrid
# import math
import os
import desc.io
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt

from mathx.core import log
from mathx.geometry import grid as gridx

from dataclasses import dataclass
import functools

# FourierRZToroidalSurface
# (m,n) -
# R(\theta,\phi)=\sum_{m,n\in M}f^R_{m,n} cos(m\theta)cos(n\zeta) where m,n<0: cos->sin
# Z(\theta,\phi)=... f^Z ...

def torus_surface(major_radius,minor_radius,NFP,mode_max,epsilon=1e-3):
  modes=[(m,n) for m in range(-1,mode_max+1) for n in range(-1,mode_max+1)]
  return desc.geometry.FourierRZToroidalSurface(
    R_lmn=[
      {
        (0,0):major_radius, #constant radius
        (1,0):minor_radius #cos(\theta)
      }.get(mod,epsilon if mod[0]>=0 else -epsilon) for mod in modes],
    modes_R=modes,
    Z_lmn=[
      {
        (-1,0):-minor_radius #sin(\theta)
      }.get(mod,epsilon if mod[0]>=0 else -epsilon) for mod in modes],
    modes_Z=modes,
    NFP=NFP
  )
  
def twisted_surface(NFP):
  return desc.geometry.FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[
      (0, 0),
      (1, 0),
      (1, 1),
      (-1, -1),
    ],  # (m,n) pairs corresponding to R_mn on previous line
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=NFP,
  )

def get_volume(eq):
  r=eq.compute("V")
  return r["V"]

@dataclass
class EquilibriumParameters:
  optimizer: str
  maxiter: int
  major_radius: float
  minor_radius: float
  NFP: int
  mode_max: int
  pressure_core: float
  iota_edge: float
  force_balance_weight: float
  quasisymmetry_two_term_weight: float
  quasisymmetry_triple_product_weight: float
  magnetic_well_weight: float

def generate_equilibrium(
  params: EquilibriumParameters):

  log.info(f"{params=}")

  surf = torus_surface(params.major_radius,
                       params.minor_radius,
                       params.NFP,
                       params.mode_max)
  # surf = twisted_surface(params.NFP.)

  pressure = desc.profiles.PowerSeriesProfile(
    # [1.8e4, 0, -3.6e4, 0, 1.8e4]
    [params.pressure_core, 0, -2*params.pressure_core, 0, params.pressure_core]
  )  # coefficients in ascending powers of rho
  iota = desc.profiles.PowerSeriesProfile([1, 0, params.iota_edge])  # 1 + 1.5 r^2
  
  # From https://github.com/PlasmaControl/DESC/blob/master/desc/examples/W7-X
  # pressure=desc.profiles.PowerSeriesProfile(
  #   [
  #     1.85596929e+05,
  #     0,
  #     -3.71193859e+05,
  #     0,
  #     1.85596929e+05,
  #     0,
  #     0
  #   ])
  # iota=desc.profiles.PowerSeriesProfile(
  #   [
  #     -8.56047021e-01,
  #     0,
  #     -3.88095412e-02,
  #     0,
  #     -6.86795128e-02,
  #     0,
  #     -1.86970315e-02,
  #     0,
  #     1.90561179e-02
  #   ])

  eq_init = desc.equilibrium.Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface=surf,
    pressure=pressure,
    iota=iota,
    Psi=1.0,  # total flux, in Webers
  )
  
  desc.plotting.plot_1d(eq_init,"p")[0].savefig("p.init.png")
  desc.plotting.plot_1d(eq_init,"iota")[0].savefig("iota.init.png")

  # eq_init, info = eq_init.solve() # Find solution with initialized field.
  # eq_init = desc.continuation.solve_continuation_automatic(eq_init,verbose=3)

  init_volume=get_volume(eq_init)
  print(f"{init_volume=}")
  
  objectives=[
    # desc.objectives.ForceBalance(eq_init),
    # desc.objectives.FusionPower(eq_init,fuel="DT"),
    # desc.objectives.Energy(eq_init),
    # desc.objectives.BoundaryError(eq_init),
    # desc.objectives.Volume(eq_init,target=get_volume(eq_init))
    # desc.objectives.Volume(eq_init,
    #                        target=init_volume),
    # desc.objectives.Elongation(eq_init,
    #                            bounds=(2,4),
    #                            weight=1),
    # desc.objectives.AspectRatio(eq=eq_init,
    #                             target=4,
    #                             weight=1),
    # desc.objectives.ForceBalance(eq=eq_init,
    #                              weight=1e6),
    # desc.objectives.Pressure(eq=eq_init,
    #                          weight=1e2),
    # desc.objectives.RotationalTransform(eq=eq_init),
    # desc.objectives.MercierStability(eq=eq_init,
    #                                  target=1,
    #                                  weight=1e-1),
  ]
  if params.quasisymmetry_two_term_weight>0:
    objectives+=[
      desc.objectives.QuasisymmetryTwoTerm(eq_init,
                                           weight=params.quasisymmetry_two_term_weight,
                                           normalize=False,
                                           helicity=(1, eq_init.NFP))]
  if params.quasisymmetry_triple_product_weight>0:
    objectives+=[
      desc.objectives.QuasisymmetryTripleProduct(eq_init,
                                                 weight=params.quasisymmetry_triple_product_weight,
                                                 normalize=False)]
  if params.magnetic_well_weight>0:
    objectives+=[
      desc.objectives.MagneticWell(eq=eq_init,
                                   bounds=(1,10),
                                   weight=params.magnetic_well_weight)]

  # eq_sol, info = eq_init.optimize(
  eq_sol, info = eq_init.solve(
    optimizer=desc.optimize.Optimizer(
      # "lsq-exact",
      # "proximal-lsq-exact",
      # "lsq-auglag",
      # "proximal-lsq-auglag",
      # "fmin-auglag-bfgs",
      # "scipy-bfgs",
      # "fmintr-bfgs",
      # "sgd",
      params.optimizer
    ),
    objective=desc.objectives.ObjectiveFunction(objectives),
    constraints=[
      desc.objectives.ForceBalance(eq=eq_init,
                                   normalize=False,
                                   weight=params.force_balance_weight),
      desc.objectives.FixPressure(eq=eq_init),
      desc.objectives.FixIota(eq=eq_init),
      desc.objectives.FixPsi(eq=eq_init),
    ],
    verbose=3,
    copy=True,
    options={
      # Sometimes the default initial trust radius is too big, allowing the
      # optimizer to take too large a step in a bad direction. If this happens,
      # we can manually specify a smaller starting radius. Each optimizer has a
      # number of different options that can be used to tune the performance.
      # See the documentation for more info.
      "initial_trust_ratio": 1.0,
      # "initial_penalty_parameter":100000,
    },
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    maxiter=params.maxiter
  )

  solution_volume=get_volume(eq_init)
  print(f"{solution_volume=}")

  return eq_sol

def generate_test_equilibrium():
  log.info("Computing equilibrium")

  surf = desc.geometry.FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[
      (0, 0),
      (1, 0),
      (1, 1),
      (-1, -1),
    ],  # (m,n) pairs corresponding to R_mn on previous line
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=5,
  )

  pressure = desc.profiles.PowerSeriesProfile(
    [1.8e4, 0, -3.6e4, 0, 1.8e4]
  )  # coefficients in ascending powers of rho
  iota = desc.profiles.PowerSeriesProfile([1, 0, 1.5])  # 1 + 1.5 r^2

  eq_init = desc.equilibrium.Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface=surf,
    pressure=pressure,
    iota=iota,
    Psi=1.0,  # total flux, in Webers
  )

  eq_sol, info = eq_init.solve(verbose=3, copy=True)

  return eq_sol

def get_test_equilibrium(path="equilibrium.h5"):
  if not os.path.exists(path):
    eq_sol=generate_test_equilibrium()

    eq_sol.save(path)

  eq=desc.io.load(path)

  return eq

# class Grid:
#   """
#   Node coordinates are in rho-phi-zeta coordinates.
#   rho is radial, phi is the major axis angle, zeta is the minor axis angle
#   The grid is is zeta, rho, phi layout (slow to fast)
#   """

#   def __init__(self,L,M,N,NFP):
#     # DESC only generates a single field period, so need to use NFP to get the values for the entire torus
#     self._grid=LinearGrid(L=L,M=M,N=N,NFP=1,sym=False,axis=True)

#   def shape(self):
#     return [self._grid.L*2,self._grid.M*2+1,self._grid.N*2+1]

#   def linear_index(self,idx):
#     shape=self.shape()
#     lidx=int(np.sum(np.array([shape[1],1,shape[1]*shape[0]])*np.array(idx)))
#     return lidx

#   def desc(self):
#     return self._grid

@dataclass
class DESCGrid:
  nodes: jnp.ndarray

  @property
  def num_nodes(self):
    return self.nodes.shape[0]

@functools.partial(jax.jit,static_argnums=(0))
def get_xyz_basis_helper(eq,rtz):
  """
  params:
    pts:
      [num_pts,3] where the points are arranged as (rho,theta,zeta) in [0,1]x[0,2*\pi]x[0,2*\pi]
  """
  # log.info("getting grid")
  grid=desc.grid.Grid(nodes=rtz,jitable=True)

  # log.info("computing values")
  r=eq.compute(["X","Y","Z",
                "X_r","X_t","X_z",
                "Y_r","Y_t","Y_z",
                "Z_r","Z_t","Z_z"],
               grid=grid,
               override_grid=False)
  # log.info("concat 1")
  xyz=jnp.concatenate([r["X"][:,None],
                       r["Y"][:,None],
                       r["Z"][:,None]],
                       axis=1)
  # log.info("concat 2")
  basis=jnp.concatenate([r["X_r"][:,None],r["X_t"][:,None],r["X_z"][:,None],
                         r["Y_r"][:,None],r["Y_t"][:,None],r["Y_z"][:,None],
                         r["Z_r"][:,None],r["Z_t"][:,None],r["Z_z"][:,None]],
                         axis=1).reshape((-1,3,3))
  return xyz,basis

def get_xyz_basis(eq,rtz):
  def callback_fn(rtz_):
    rtz__=rtz_[None,...] if len(rtz_.shape)==1 else rtz_
    xyz,basis=get_xyz_basis_helper(eq,rtz__)
    xyz,basis=(xyz[0],basis[0]) if len(rtz_.shape)==1 else (xyz,basis)
    xyz.block_until_ready()
    basis.block_until_ready()
    return xyz,basis

  return jax.pure_callback(
    callback_fn,
    (jax.ShapeDtypeStruct((3,),rtz.dtype),
     jax.ShapeDtypeStruct((3,3,),rtz.dtype)),
    rtz,
    vmap_method="expand_dims")

###############################################################################
# get_u

@functools.partial(jax.jit,static_argnums=(0))
def get_u_helper(eq,x):
  """
  The internal map_coordinates implementation makes use of numpy so we can't
  jit it. So it is generally very slow. Use the scipy minimize functionality
  instead.
  """
  # grid=desc.grid.Grid(nodes=x)
  # rtz=eq.map_coordinates(x,inbasis=("X","Y","Z"),outbasis=("rho","theta","zeta"))

  eps=.001
  num_nodes=x.shape[0]

  # Pick a starting point at the same zeta and slightly offset from the
  # equilibrium center. This may be singular if the gradient path leads
  # back to the center, in which case the TODO is to implement a random
  # initialization retry.
  rtz_init=jnp.concatenate([jnp.zeros((num_nodes,1))+eps,
                            jnp.zeros((num_nodes,1)),
                            jnp.arctan2(x[:,1],x[:,0])[...,None]],
                        axis=1)

  def compute_xyz_fn(rtz):
    x=eq.compute(["X","Y","Z"],
                 desc.grid.Grid(nodes=rtz,jitable=True),
                 override_grid=False)
    x=jnp.concatenate([x["X"][...,None],
                       x["Y"][...,None],
                       x["Z"][...,None]],
                      axis=1)
    return x

  def obj_fn(rtz):
    rtz=rtz.reshape(-1,3)
    return jnp.linalg.norm(compute_xyz_fn(rtz)-x)**2

  rtz=jopt.minimize(obj_fn,rtz_init.reshape(-1),method="BFGS").x.reshape(-1,3)
  return rtz

# @jax.custom_batching.custom_vmap
# def get_u(eq,x):
#   return get_u_helper(eq,x[None,...])[0]

# @get_u.def_vmap
# def get_u_vmap(axis_size,in_batched,eq,x):
#   return get_u_helper(eq,x),True

def get_u(eq,x):
  def callback_fn(x_):
    x__=x_[None,...] if len(x_.shape)==1 else x_
    u=get_u_helper(eq,x__)
    u=u[0] if len(x_.shape)==1 else u
    u.block_until_ready()
    return u

  return jax.pure_callback(
    callback_fn,
    (jax.ShapeDtypeStruct((3,),x.dtype)),
    x,
    vmap_method="expand_dims")

###############################################################################
# get_B

@functools.partial(jax.jit,static_argnums=(0))
def get_B_helper(eq,u):
  num_nodes=u.shape[0]
  rtz=u
  grid=desc.grid.Grid(nodes=rtz,
                      jitable=True,
                      sort=False,
                      is_meshgrid=False,
                      spacing=jnp.ones((num_nodes,3)),
                      weights=jnp.ones((num_nodes,)))
  # xyz=

  # r=eq.compute(["B","R","phi","Z"],
  #              grid=grid,
  #              override_grid=False)

  # rpz=jnp.concatenate([r["R"][:,None],
  #                      r["phi"][:,None],
  #                      r["Z"][:,None]],
  #                     axis=1)

  # B=r["B"]
  # B=jax.vmap(gridx.convert_cylindrical_to_cartesian,in_axes=(0,0),out_axes=(0))(rpz,B)

  r=eq.compute(["B^rho","B^theta","B^zeta"],
               grid,
               override_grid=False)
  B_rtz=jnp.concatenate([r["B^rho"][...,None],
                         r["B^theta"][...,None],
                         r["B^zeta"][...,None]],
                        axis=1)

  return B_rtz

  #print(r.keys())
  # B=jax.vmap(gridx.convert_cylindrical_to_cartesian_vector,in_axes=(0,0),out_axes=(0))(grid.r["B"]

# @jax.custom_batching.custom_vmap
# def get_B(eq,u):
#   return get_B_helper(eq,u[None,...])[0]

# @get_B.def_vmap
# def get_B_vmap(axis_size,in_batched,eq,u):
#   return get_B_helper(eq,u)

def get_B(eq,u):
  def callback_fn(u_):
    u__=u_[None,...] if len(u_.shape)==1 else u_
    B=get_B_helper(eq,u__)
    B=B[0] if len(u_.shape)==1 else B
    B.block_until_ready()
    return B

  return jax.pure_callback(
    callback_fn,
    (jax.ShapeDtypeStruct((3,),u.dtype)),
    u,
    vmap_method="expand_dims")
