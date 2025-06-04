from . import hexagon
from mathx.core import log
import numpy as np
import jax
import jax.numpy as jnp

def remove_degenerate(idx_arr):
  return [ix for ix in idx_arr if len(set(ix))==len(ix)]

def compute_degenerate_idx(idx,vshape,degenerate):
  """
  See generate_mesh_volume
  """
  # print(degenerate)
  real_idx=list(idx)
  if degenerate is not None:
    for d in range(3):
      if degenerate[d] is not None:
        for e in range(2):
          if idx[d]==(e*(vshape[d]-1)) and degenerate[d][e] is not None:
            for j in range(2):
              if degenerate[d][e][j]:
                # print(f"{d} {e} {j}")
                real_idx[(d+j+1)%3]=0
  # if idx!=real_idx:
  #   print(f"{idx} {real_idx}")
  return tuple(real_idx)

def generate_mesh_volume(pos_fn,segments,closed=(False,False,False),degenerate=None):
  """
  Generates a volumetric tetrahedral mesh for a given function defining the mesh positions. The mesh can then be closed at end pairs, or degenerate at ends in either/both tangential axes.

  params:
    pos_fn: The position function [0,1]^3 -> R^3.
    segments: The number of segments in each dimension.
    closed: Whether each end loops back on itself.
    degenerate: A 3-tuple where for each dimension, a 2-tuple contains for each end, a 2 tuple indicates whether each tangential axis is degenerate/collapsed to a single point.
  """
  D=3

  sshape=segments
  # for i in range(D):
  #   # assert sshape[i]%2==0 or closed[i]==False
  sshape=tuple([(s+1 if (closed[i] and s%2==1) else s) for i,s in enumerate(sshape)])

  # vtx_arr=np.ndarray([num_toroidal,num_poloidal,num_radial+1,3])

  vshape=tuple([n+1 if closed[i]==False else n for i,n in enumerate(segments)])
  # print(f"vshape={vshape} sshape={sshape}")

  vtx=[]
  vtx_idx=np.ndarray(vshape)
  vtx_idx.fill(-1)
  for idx in np.ndindex(vshape):
    real_idx=compute_degenerate_idx(idx,vshape,degenerate)
    if vtx_idx[real_idx]==-1:
      vtx_idx[real_idx]=len(vtx)
      uvw=np.array(real_idx,dtype=np.float32)/np.array(sshape)
      p=pos_fn(uvw)
      # print(f"uvw={uvw} p={p}")
      vtx.append(p)
    vtx_idx[idx]=vtx_idx[real_idx]

  def wrap(*ix):
    return tuple([n%vshape[i] for i,n in enumerate(ix)])

  tet_idx=[]
  for idx in np.ndindex(sshape):
    tet_idx.extend(remove_degenerate(
      hexagon.hexagon_to_tetrahedron([
        vtx_idx[wrap(idx[0],  idx[1],  idx[2]  )],
        vtx_idx[wrap(idx[0]+1,idx[1],  idx[2]  )],
        vtx_idx[wrap(idx[0],  idx[1]+1,idx[2]  )],
        vtx_idx[wrap(idx[0]+1,idx[1]+1,idx[2]  )],
        vtx_idx[wrap(idx[0],  idx[1],  idx[2]+1)],
        vtx_idx[wrap(idx[0]+1,idx[1],  idx[2]+1)],
        vtx_idx[wrap(idx[0],  idx[1]+1,idx[2]+1)],
        vtx_idx[wrap(idx[0]+1,idx[1]+1,idx[2]+1)]
      ],reverse=sum(idx)%2==1)))
  # print(vtx_idx)
  # print(tet_idx)

  return vtx,tet_idx

class SimplexMesh:
  def __init__(self):
    self.vertex=[]
    self.element=[]

  def subdivide(mesh,n):
    #for each element, loop over d-1 codimensions, generate nodes if they don't exist, then add tesselated sub-elements
    pass

def generate_mesh_surface(pos_fn,segments,closed=(False,False,False),degenerate=None):
  """
  params:
    See generate_mesh_volume

  TODO:
    Make this differentiable by instead returning the local space points. Then we can get the derivative by JAX pos_fn and then taking derivatives with respect to the local space points.
  """
  sshape=segments

  vshape=tuple([n+1 if closed[i]==False else n for i,n in enumerate(segments)])

  vtx_u=[]
  vtxm={} #array index to linear idx
  elem=[]

  D=3

  log.info("generating topology")
  for d in range(D):
    for e in range(2):
      if not closed[d]:
        for fidx in np.ndindex(tuple([v for i,v in enumerate(sshape) if i!=d])):
          el=[]
          for voff in np.ndindex(tuple([2]*(D-1))):
            idx=np.array(fidx)+np.array(voff)
            # print(idx)
            idx=tuple(np.concatenate([idx[:d],[e*sshape[d]],idx[d:]]).tolist())
            # print(fidx,voff,idx)
            real_idx=compute_degenerate_idx(idx,vshape,degenerate)
            # if real_idx!=idx:
            #   print(idx,real_idx)
            if idx not in vtxm:
              uvw=np.array(real_idx,dtype=np.float32)/np.array(sshape)
              # print(real_idx,sshape,uvw)
              vtxm[idx]=len(vtx_u)
              vtx_u.append(uvw)
            vi=vtxm[real_idx]
            el.append(vi)
          #TODO: if e==1, reverse element vertex order
          if e==1:
            el=[e for e in reversed(el)]

          # print(el)

          # elem.append[d]
          elem.extend(remove_degenerate(
            [(el[0],el[1],el[2]),
             (el[2],el[1],el[3])]))

  vtx_u=jnp.array(vtx_u)
  batch_pos_fn=jax.vmap(pos_fn,in_axes=(0),out_axes=(0))
  # batch_pos_fn=jax.jit(batch_pos_fn)
  vtx=batch_pos_fn(vtx_u)

  #Convert back to array
  vtx=[v for v in vtx]

  return vtx,elem

def surface_basis_transform(pos_fn):
  """
  Transform a surface function f(x,y) into a function that returns the normal:
  \vec{xs}=pos_fn(u,v)

  du=\frac{\partial\vec{xs}}{u}
  dv=\frac{\partial\vec{xs}}{u}
  \vec{n}=\frac{d0\times d1}{|d0||d1|}

  """
  pos_jac_fn=jax.jacrev(pos_fn)
  def fn(u):
    x=pos_fn(u)
    dx=pos_jac_fn(u)
    # print(dx)
    du=dx[:,0]
    dv=dx[:,1]
    n=jnp.cross(du,dv)
    n=n/jnp.linalg.norm(n)
    return n
  return fn

def contravariant_basis_transform(pos_fn):
  pos_jac_fn=jax.jacrev(pos_fn)
  return pos_jac_fn

def surface_normal_transform(pos_fn):
  """
  Transform a surface function f(x,y) into a function that returns the normal:
  \vec{xs}=pos_fn(u,v)

  du=\frac{\partial\vec{xs}}{u}
  dv=\frac{\partial\vec{xs}}{u}
  \vec{n}=\frac{d0\times d1}{|d0||d1|}

  """
  dx_fn=contravariant_basis_transform(pos_fn)
  def fn(x):
    dx=dx_fn(x)
    du=dx[:,0]
    dv=dx[:,1]
    n=jnp.cross(du,dv)
    n=n/jnp.linalg.norm(n)
    return n
  return fn

class Curvilinear:
  D: int = 3

  def __init__(self,
               pos_fn: callable,
               closed=(False,False,False),
               degenerate=(None,None,None),
               min_segments=(1,1,1)):
    self.pos_fn=pos_fn
    self.closed=closed
    self.degenerate=degenerate
    self.min_segments=min_segments
    self.batch_pos_fn=jax.vmap(self.pos_fn,in_axes=(0),out_axes=(0))
    # self.batch_pos_fn=jax.jit(self.batch_pos_fn)

  def compute_density(self,num):
    log.info("compute_density")

    num_steps=16
    ls=np.array([0.]*self.D)
    angles=np.array([0.]*self.D)
    for d in range(self.D):
      p=np.array([0.5]*self.D)

      us=.5*jnp.ones([num_steps,2])
      u_steps=jnp.linspace(0,1,num_steps,endpoint=False if self.closed[d] else True)
      us=jnp.concatenate([us[:,:d],u_steps[:,None],us[:,d:]],axis=1)
      # log.info(f"density {us=}")
      xs=self.batch_pos_fn(us)

      #TODO: vectorize

      # Distance calculation
      # l=0
      # for i in range(num_steps-1):
      #   dx=jnp.linalg.norm(xs[i+1]-xs[i])
      #   l+=dx
      # ls[d]=l

      # Angle calculation
      a=0
      for i in range(num_steps-(0 if self.closed[d] else 2)):
        dx1=xs[(i+2)%num_steps]-xs[(i+1)%num_steps]
        dx0=xs[(i+1)%num_steps]-xs[i]
        da=jnp.acos(jnp.clip((dx1 @ dx0)/(jnp.linalg.norm(dx1)*jnp.linalg.norm(dx0)),min=-1,max=1))
        if jnp.isnan(da).any():
          import pdb
          pdb.set_trace()
        a+=da
      angles[d]=a

    # Distance calculation
    # segments=tuple(jnp.maximum(1,jnp.round(num*ls/jnp.max(ls))).astype(int).tolist())
    # print(f"{ls=} {segments=} {num=}")

    # Angle calculation
    segments=tuple(jnp.maximum(1,jnp.round(num*angles/(2*jnp.pi))).astype(int).tolist())
    log.info(f"{angles=} {segments=} {num=}")

    return segments

  def tesselate_volume(self,density):
    segments=self.compute_density(density)
    if self.min_segments:
      segments=tuple([max(self.min_segments[d],s) for d,s in enumerate(segments)])
    return generate_mesh_volume(
      self.pos_fn,
      segments,
      closed=self.closed,
      degenerate=self.degenerate)

  def tesselate_surface(self,density):
    segments=self.compute_density(density)
    return generate_mesh_surface(
      self.pos_fn,
      segments,
      closed=self.closed,
      degenerate=self.degenerate)
