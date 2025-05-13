from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.continuation import solve_continuation_automatic
from desc.grid import LinearGrid
import math
import os
import desc.io
import numpy as np
import jax.numpy as jnp

from mathx.core import log

def generate_test_equilibrium():
  log.info("Computing equilibrium")  

  surf = FourierRZToroidalSurface(
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

  pressure = PowerSeriesProfile(
    [1.8e4, 0, -3.6e4, 0, 1.8e4]
  )  # coefficients in ascending powers of rho
  iota = PowerSeriesProfile([1, 0, 1.5])  # 1 + 1.5 r^2

  eq_init = Equilibrium(
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

class Grid:
  """
  Node coordinates are in rho-phi-zeta coordinates.
  rho is radial, phi is the major axis angle, zeta is the minor axis angle
  The grid is is zeta, rho, phi layout (slow to fast)
  """

  def __init__(self,L,M,N,NFP):
    # DESC only generates a single field period, so need to use NFP to get the values for the entire torus
    self._grid=LinearGrid(L=L,M=M,N=N,NFP=1,sym=False,axis=True)

  def shape(self):
    return [self._grid.L*2,self._grid.M*2+1,self._grid.N*2+1]

  def linear_index(self,idx):
    shape=self.shape()
    lidx=int(np.sum(np.array([shape[1],1,shape[1]*shape[0]])*np.array(idx)))
    return lidx

  def desc(self):
    return self._grid

def get_xyz_basis(eq,u):
  """
  params:
    pts:
      [num_pts,3] where the points are arranged as (phi,theta,rho)
      desc takes points in (rho,theta,zeta) (where phi=zeta) so we have to reverse the order of the points.
  """

  # pts_desc=pts
  # if len(pts.shape)==1
  #   pts_desc=pts_desc[None]
  # pts_desc=jnp.flip(pts,axis=-1)
  # pts=jnp.array([u])

  # grid = Grid(L=1,M=32,N=32,NFP=eq.NFP)
  # xyz=eq.compute(["X","Y","Z"],grid=grid.desc())

  rtz=u[:,::-1]*jnp.array([[1,2*jnp.pi,2*jnp.pi]])
  grid=desc.grid.Grid(nodes=rtz)
  r=eq.compute(["X","Y","Z",
                "X_r","X_t","X_z",
                "Y_r","Y_t","Y_z",
                "Z_r","Z_t","Z_z"],grid=grid)  
  xyz=jnp.concatenate([r["X"][:,None],
                       r["Y"][:,None],
                       r["Z"][:,None]],
                       axis=1)
  basis=jnp.concatenate([r["X_r"][:,None],r["X_t"][:,None],r["X_z"][:,None],
                         r["Y_r"][:,None],r["Y_t"][:,None],r["Y_z"][:,None],
                         r["Z_r"][:,None],r["Z_t"][:,None],r["Z_z"][:,None]],
                         axis=1).reshape((-1,3,3))
  # print(rtz)
  # print(xyz)
  return xyz,basis

  # def remap_desc(grid,field):
  #   shape=grid.shape()
  #   arr=np.ndarray(shape)
  #   for k in range(shape[2]):
  #     for i in range(shape[0]):
  #       for j in range(shape[1]):
  #         idx0=grid.linear_index([i,j,k])
  #         arr[i,j,k]=field[idx0]
  #   return arr

  # grid_xyz=np.concat([remap_desc(grid,[xyz_split["X"])[...,None],
  #                     remap_desc(grid,xyz_split["Y"])[...,None],
  #                     remap_desc(grid,xyz_split["Z"])[...,None]],
  #                    axis=-1)
  # print(grid_xyz)
  # print(grid_xyz.shape)
  
  # return grid_xyz
