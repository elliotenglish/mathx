from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.continuation import solve_continuation_automatic
from desc.grid import LinearGrid
import math
import os
import desc.io
import numpy as np
import scipy

def generate():
  eq_path="equilibrium.h5"
  plot_path="fig.html"

  if not os.path.exists(eq_path):
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

    # eq = solve_continuation_automatic(eq_sol.copy(), verbose=3)
    eq=eq_sol

    eq.save(eq_path)

  eq=desc.io.load(eq_path)

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

  grid = Grid(L=1,M=32,N=32,NFP=eq.NFP)
  xyz=eq.compute(["X","Y","Z"],grid=grid.desc())

  def remap_desc(grid,field):
    shape=grid.shape()
    arr=np.ndarray(shape)
    for k in range(shape[2]):
      for i in range(shape[0]):
        for j in range(shape[1]):
          idx0=grid.linear_index([i,j,k])
          arr[i,j,k]=field[idx0]
    return arr

  grid_xyz=np.concat([remap_desc(grid,xyz["X"])[...,None],
                      remap_desc(grid,xyz["Y"])[...,None],
                      remap_desc(grid,xyz["Z"])[...,None]],
                    axis=-1)
  print(grid_xyz.shape)

  # print(xyz)
  # print(xyz.shape)

  def generate_surface(grid_xyz,closed_xyz):
    pts=[]
    tris=[]

    shape=grid_xyz.shape[:3]

    idxs=np.ndarray(shape)

    #Store all the points so as to not break the indexing
    for i in range(shape[0]):
      for j in range(shape[1]):
        for k in range(shape[2]):
          idxs[i,j,k]=len(pts)
          pts.append(grid_xyz[i,j,k])

    r=-1
    for i in range(shape[1] if closed_xyz[1] else shape[1]-1):
      for j in range(shape[2] if closed_xyz[2] else shape[2]-1):
        tris.append((idxs[r,i,j],
                    idxs[r,(i+1)%shape[1],j],
                    idxs[r,(i+1)%shape[1],(j+1)%shape[2]]))
        tris.append((idxs[r,i,j],
                    idxs[r,(i+1)%shape[1],(j+1)%shape[2]],
                    idxs[r,i,(j+1)%shape[2]]))

    return pts,tris

  plasma_pts,plasma_tris=generate_surface(grid_xyz,[False,True,True])

  def generate_cylinder(center,axis,length,radius):
    shape=[1,2,32,3]
    grid_xyz=np.ndarray(shape)
    translation=center
    cross=np.cross(axis,[0,0,1])
    cross_norm=np.linalg.norm(cross)
    angle=math.asin(cross_norm)
    rotation=scipy.spatial.transform.Rotation.from_rotvec(cross*angle/cross_norm)
    for i in range(shape[1]):
      for j in range(shape[2]):
        z=float(i)/(shape[1]-1)*length
        angle=float(j)/shape[2]*2*math.pi
        grid_xyz[0,i,j]=np.array([math.cos(angle),math.sin(angle),z])*radius
        grid_xyz[0,i,j]=translation+rotation.as_matrix()@grid_xyz[0,i,j]
    return grid_xyz

  magnets=[]
  nfp=5
  for i in range(nfp):
    angle=math.pi*2*i/nfp
    length=4
    radius=2
    dir=np.array([-math.sin(angle),math.cos(angle),0])
    x=np.array([math.cos(angle),math.sin(angle),0])*10+dir*length
    cyl_pts,cyl_tris=generate_surface(generate_cylinder(x,dir,length,radius),[False,False,True])
    magnets.append([cyl_pts,cyl_tris])

  import code_fusion_reactor.visualization as viz

  elements=[]
  elements.append(viz.generate_mesh3d(plasma_pts,plasma_tris,color=[255,0,0]))
  for m in magnets:
    elements.append(viz.generate_mesh3d(m[0],m[1],color=[0,255,0]))
  viz.write_visualization(elements,"fig.html")
  