from mathx.geometry import tetrahedron
import pytest

def test_tetrahedron():
  points=[[0,0,0],[1,0,0],[0,0,1],[0,1,0]]
  
  size=tetrahedron.tetrahedron_signed_volume(*points)
  assert size==pytest.approx(1./6.,abs=1e-5)
 
  points_rewound=points[1:]+points[:1]
  size_rewound=tetrahedron.tetrahedron_signed_volume(*points_rewound)
  assert size_rewound==-size
  
  print(f"points={points}")
  print(f"size={size}")
