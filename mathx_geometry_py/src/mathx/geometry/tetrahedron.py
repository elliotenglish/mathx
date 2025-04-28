import numpy as np

def tetrahedron_to_triangle(t):
  return [(t[0],t[1],t[2]),
          (t[1],t[3],t[2]),
          (t[2],t[3],t[0]),
          (t[3],t[1],t[0])]

def tetrahedron_signed_volume(v1, v2, v3, v4):
  """
  Calculates the volume of a tetrahedron.

  Args:
      v1, v2, v3, v4: Coordinates of the four vertices as lists or numpy arrays.

  Returns:
      The volume of the tetrahedron.
  """
  # matrix = np.array([
  #     [v1[0], v1[1], v1[2], 1],
  #     [v2[0], v2[1], v2[2], 1],
  #     [v3[0], v3[1], v3[2], 1],
  #     [v4[0], v4[1], v4[2], 1]
  # ])
  matrix=np.concatenate([np.array([v1,v2,v3,v4]).T,[[1,1,1,1]]])
  return np.linalg.det(matrix) / 6
