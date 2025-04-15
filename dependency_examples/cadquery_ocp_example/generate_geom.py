"""
To view the output use:
meshlab body.stl
"""

import OCP

p0=OCP.gp.gp_Pnt(0,0,0)
p1=OCP.gp.gp_Pnt(1,1,0)
p2=OCP.gp.gp_Pnt(2,0,0)

# Define 2d geometry elements
arc=OCP.GC.GC_MakeArcOfCircle(p0,p1,p2).Value()
segment=OCP.GC.GC_MakeSegment(p2,p0).Value()

# Define 2d geometry composite elements
arc_edge=OCP.BRepBuilderAPI.BRepBuilderAPI_MakeEdge(arc).Edge()
segment_edge=OCP.BRepBuilderAPI.BRepBuilderAPI_MakeEdge(segment).Edge()

wire_builder=OCP.BRepBuilderAPI.BRepBuilderAPI_MakeWire()
wire_builder.Add(arc_edge)
wire_builder.Add(segment_edge)
wire=wire_builder.Wire()

face=OCP.BRepBuilderAPI.BRepBuilderAPI_MakeFace(wire).Face()
prism_vec=OCP.gp.gp_Vec(0,0,.5)

body=OCP.BRepPrimAPI.BRepPrimAPI_MakePrism(face,prism_vec).Prism()
print(body)

shape=body.Shape()

mesh = OCP.BRepMesh.BRepMesh_IncrementalMesh(shape, 0.1)
mesh.Perform()

writer=OCP.StlAPI.StlAPI_Writer()
writer.ASCIIMode=True
error=writer.Write(shape,"body.stl")
print(dir(writer))
print(error)
assert error

# writer=OCP.STEPControl.STEPControl_Writer()
# writer.Transfer(body.Shape(),OCP.STEPControl.STEPControl_AsIs)
# fileStream=OCP.OSD.OSD_OStream("body.step")
# writer.write(fileStream)
# dir(writer)
# dir(fileStream)
