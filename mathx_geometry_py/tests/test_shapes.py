import mathx.geometry.torus as torus
from mathx.geometry.torus import Torus
from mathx.geometry.cylinder import Cylinder
from mathx.geometry.box import Box
import mathx.geometry.visualization as viz
import numpy as np

def shape_test_helper(name,shape):
  print()
  print(name)

  vtx,tet_idx=shape.tesselate_volume(16)
  print(f"{len(vtx)=} {len(tet_idx)=}")
  tri_idx=viz.get_boundary_faces(tet_idx)

  # vtx,tri_idx=shape.tesselate_surface(16)

  print(f"{len(tri_idx)=}")
  # print(vtx)
  # print(tet_idx)
  # print(tri_idx)
  viz.write_visualization(
    [
      viz.generate_mesh3d(vtx,
                          tri_idx,
                          color=[255,0,0],
                          wireframe=True,
                          flatshading=True),
      # viz.generate_scatter3d(vtx,color=[0,0,255])
    ],
    f"{name}.html")

def test_torus_hollow():
  shape_test_helper("Torus.hollow",Torus(major_radius=2,minor_radius_inner=.5,minor_radius_outer=1))

def test_torus_solid():
  shape_test_helper("Torus.solid",Torus(major_radius=2,minor_radius_inner=0,minor_radius_outer=.2))
  
def test_torus_toroid():
  major=1.5
  minor=.25
  rtp=(.24,.543,.879)
  xyz=torus.toroid_to_xyz(major,0,minor,*rtp)
  rtp0=torus.xyz_to_toroid(major,0,minor,*xyz)
  np.testing.assert_allclose(np.array(rtp),np.array(rtp0),rtol=1e-5)

def test_cylinder_hollow():
  shape_test_helper("Cylinder.hollow",Cylinder(radius=.1,length=.05,thickness=.2))
  
def test_cylinder_solid():
  shape_test_helper("Cylinder.solid",Cylinder(radius=0,length=.6,thickness=.2))
  
def test_box():
  shape_test_helper("Box",Box(length=[1,1,1]))
