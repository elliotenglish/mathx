import mathx.geometry.tetrahedron as tetrahedron
import mathx.geometry.hexagon as hexagon
import numpy as np
import pytest

def test_hexagon():
  vtx=[[0,0,0],[1,0,0],[0,1,0],[1,1,0],
       [0,0,1],[1,0,1],[0,1,1],[1,1,1]]

  tet_idxs_arr=hexagon.hexagon_to_tetrahedron_arr([[[0,1],[2,3]],[[4,5],[6,7]]])
  print(tet_idxs_arr)
  
  tet_idxs=hexagon.hexagon_to_tetrahedron([0,1,2,3,4,5,6,7])
  print(tet_idxs)
  
  assert(tet_idxs==tet_idxs_arr)
  assert(len(tet_idxs)==5)

  vol_sum=0
  for ti in tet_idxs:
    vol=tetrahedron.tetrahedron_signed_volume(vtx[ti[0]],vtx[ti[1]],vtx[ti[2]],vtx[ti[3]])
    print(vol)
    assert vol>0
    vol_sum+=vol
  assert vol_sum==pytest.approx(1,abs=1e-5)
