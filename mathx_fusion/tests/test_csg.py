from mathx.geometry import visualization as viz
from mathx.geometry import hexagon as hexagon
from mathx.core import log
import logging

# from CGAL import CGAL_Triangulation_3 as T3
from CGAL.CGAL_Kernel import Point_3 as P3
import CGAL.CGAL_Mesh_3 as M3

# def make_cube_3(P):
#     # appends a cube of size [0,1]^3 to the polyhedron P.
#     assert P.is_valid()
#     h = P.make_tetrahedron(Point_3(1, 0, 0), Point_3(0, 0, 1),
#                            Point_3(0, 0, 0), Point_3(0, 1, 0))
#     g = h.next().opposite().next()
#     P.split_edge(h.next())
#     P.split_edge(g.next())
#     P.split_edge(g)
#     h.next().vertex().set_point(Point_3(1, 0, 1))
#     g.next().vertex().set_point(Point_3(0, 1, 1))
#     g.opposite().vertex().set_point(Point_3(1, 1, 0))
#     f = P.split_facet(g.next(), g.next().next().next())
#     e = P.split_edge(f)
#     e.vertex().set_point(Point_3(1, 1, 1))
#     P.split_facet(e, f.next().next())
#     assert P.is_valid()
#     return h
  
from CGAL.CGAL_Triangulation_3 import Triangulation_3
from CGAL.CGAL_Mesh_3 import make_mesh_3, Default_mesh_criteria
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Point_3

def construct_cgal_mesh(points, tetrahedra):
  # Create a Delaunay triangulation
  triangulation = Triangulation_3()
  
  # Insert points into the triangulation
  point_handles = [triangulation.insert(Point_3(*p)) for p in points]
  
  # Insert predefined tetrahedral cells manually
  for t in tetrahedra:
      triangulation.tds().insert_cell(
          point_handles[t[0]],
          point_handles[t[1]],
          point_handles[t[2]],
          point_handles[t[3]]
      )
  
  # Define mesh criteria (adjust parameters as needed)
  criteria = Default_mesh_criteria(
      facet_angle=30,
      facet_size=0.1,
      facet_distance=0.05,
      cell_radius_edge_ratio=3.0,
      cell_size=0.1
  )
  
  # Generate mesh
  mesh = make_mesh_3(triangulation, criteria)
  
  # Convert to polyhedron for easy output
  # polyhedron = Polyhedron_3()
  # triangulation.convert_to_polyhedron(polyhedron)
  
  # return polyhedron  
  return triangulation

def make_box():
  vtx=[(0,0,0),(1,0,0),(0,1,0),(1,1,0),
       (0,0,1),(1,0,1),(0,1,1),(1,1,1)]
  
  #idx=hexagon.hexagon_to_tetrahedron([i for i in range(8)])
  idx=[(0,1,2,3)]

  return construct_cgal_mesh(vtx,idx)
  
  # for v in vtx:
  #   mesh.push_back_vertex(v)
    
  # for i in idx:
  #   mesh.push_back_cell(*i)

  # return mesh

def test_csg():
  initialize_logging()
  
  # logging.info([(p.x(),p.y(),p.z()) for p in cgal_points])

  #Create a box [0,1]
  #Create a box [.5,1.5] by translating the first box
  
  #Subtract the second box from the first

  #Visualize the box

  box=make_box()
  print(box)
