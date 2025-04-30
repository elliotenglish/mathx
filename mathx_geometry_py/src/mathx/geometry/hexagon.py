import numpy as np

def to_type(arr,tp):
  return [[tp(a) for a in ar] for ar in arr]

def hexagon_to_tetrahedron_arr(vidxs):
  r"""
  Vertices must be ordered as follows:
  0:[0,0,0] -- 1:[1,0,0]
    |    \        |   \
    |     \       |    \
    |      \      |     \
    |  4:[0,0,1] -+ 5:[1,0,1]
    |       |     |      |
  2:[0,1,0] +- 3:[1,1,0] |
     \      |      \     |
      \     |       \    |
       \    |        \   |
       6:[0,1,1] -- 7:[1,1,1]
  
  TODO: This will not yield a valid mesh in general as neighboring hexahedra need to have tetrahedra generated independently in order to have matching face tesselation. This will likely require incremental construction in order to achieve a consistent tesselation. An ear clipping approach is likely sufficient for convex hexahedra.
  """
  vidxs=np.array(vidxs)
  # print(vidxs)

  # return to_type([[vidxs[0,0,0],vidxs[1,0,0],vidxs[0,1,0],vidxs[0,0,1]],
  #                 [vidxs[1,1,0],vidxs[0,1,0],vidxs[1,0,0],vidxs[1,1,1]],
  #                 [vidxs[0,1,1],vidxs[1,1,1],vidxs[0,0,1],vidxs[0,1,0]],
  #                 [vidxs[1,0,1],vidxs[0,0,1],vidxs[1,1,1],vidxs[1,0,0]],
  #                 [vidxs[1,0,0],vidxs[0,1,0],vidxs[0,0,1],vidxs[1,1,1]]],int)

  return to_type([[vidxs[0,0,0],vidxs[1,0,0],vidxs[0,1,0],vidxs[0,0,1]],
                  [vidxs[1,1,0],vidxs[0,1,0],vidxs[1,0,0],vidxs[1,1,1]],
                  [vidxs[0,1,1],vidxs[1,1,1],vidxs[0,0,1],vidxs[0,1,0]],
                  [vidxs[1,0,1],vidxs[0,0,1],vidxs[1,1,1],vidxs[1,0,0]],
                  [vidxs[1,0,0],vidxs[0,1,0],vidxs[0,0,1],vidxs[1,1,1]]],int)

def hexagon_to_tetrahedron(vidxs):
  r"""
  Vertices must be ordered as follows:
  0:[0,0,0] -- 1:[1,0,0]
    |    \        |   \
    |     \       |    \
    |      \      |     \
    |  4:[0,0,1] -+ 5:[1,0,1]
    |       |     |      |
  2:[0,1,0] +- 3:[1,1,0] |
     \      |      \     |
      \     |       \    |
       \    |        \   |
       6:[0,1,1] -- 7:[1,1,1]
  """
  vidxs=np.array(vidxs)
  return hexagon_to_tetrahedron_arr(vidxs.reshape([2,2,2]))
