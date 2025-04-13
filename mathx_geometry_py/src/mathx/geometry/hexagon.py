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
