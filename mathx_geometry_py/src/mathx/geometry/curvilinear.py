from . import hexagon
import numpy as np

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

def generate_mesh_volume(posfn,segments,closed=(False,False,False),degenerate=None):
  """
  Generates a volumetric tetrahedral mesh for a given function defining the mesh positions. The mesh can then be closed at end pairs, or degenerate at ends in either/both tangential axes.

  params:
    posfn: The position function [0,1]^3 -> R^3.
    segments: The number of segments in each dimension.
    closed: Whether each end loops back on itself.
    degenerate: A 3-tuple where for each dimension, a 2-tuple contains for each end, a 2 tuple indicates whether each tangential axis is degenerate/collapsed to a single point.
  """
  sshape=segments

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
      p=posfn(uvw)
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
      ])))
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

def generate_mesh_surface(posfn,segments,closed=(False,False,False),degenerate=None):
  """
  params:
    See generate_mesh_volume
  """
  sshape=segments

  vshape=tuple([n+1 if closed[i]==False else n for i,n in enumerate(segments)])
  
  vtx=[]
  vtxm={} #array index to linear idx
  elem=[]
  
  D=3

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
              vtxm[idx]=len(vtx)
              vtx.append(posfn(uvw))
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
  return vtx,elem
