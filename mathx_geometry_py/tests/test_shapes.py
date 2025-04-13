from mathx.geometry.torus import Torus
from mathx.geometry.cylinder import Cylinder
import mathx.geometry.visualization as viz

def shape_test_helper(name,shape):
  print()
  print(name)
  vtx,tet_idx=shape.tesselate_volume(8)
  print(f"vtx={len(vtx)} tet_idx={len(tet_idx)}")
  tri_idx=viz.tetrahedron_to_triangle(tet_idx)
  # print(vtx)
  # print(tet_idx)
  # print(tri_idx)
  viz.write_visualization(
    [
      viz.generate_mesh3d(vtx,tri_idx,color=[255,0,0]),
      # viz.generate_scatter3d(vtx,color=[0,0,255])
      ],
    f"{name}.html")

def test_torus_hollow():
  shape_test_helper("Torus.mr05",Torus(major_radius=2,minor_radius=.05,thickness=.2))

def test_torus():
  shape_test_helper("Torus.mr0",Torus(major_radius=2,minor_radius=0,thickness=.2))

def test_cylinder_hollow():
  shape_test_helper("Cylinder.r0001",Cylinder(radius=.1,length=.05,thickness=.2))
  
def test_cylinder():
  shape_test_helper("Cylinder.r0",Cylinder(radius=0,length=.05,thickness=.2))
