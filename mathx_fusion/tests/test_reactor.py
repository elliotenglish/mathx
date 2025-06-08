import mathx.fusion.reactor as freact
import mathx.fusion.torus_plasma as ftplasma
import mathx.fusion.stellarator_plasma as fsplasma
import mathx.geometry.visualization as viz
from mathx.core import log

import numpy as np

def clip_mesh(mesh):
  verts=[]
  vmap=dict()
  for i,v in enumerate(mesh[0]):
    if v[0]<1:
      vmap[i]=len(verts)
      verts.append(v)

  tris=[[vmap[vi] for vi in t] for t in mesh[1] if all([vi in vmap for vi in t])]
  return verts,tris

def reactor_test_helper(plasma,prefix=""):
  rand=np.random.default_rng(54323)
  
  log.info("creating reactor")
  params=freact.ReactorParameters(
    wall_thickness=.001,
    magnets_conformal_num=0,
    # magnets_conformal_num=7*plasma.nfp,
    magnets_conformal_width=.2,
    magnets_ring_num=1,
    # magnets_ring_num=3*plasma.nfp,
    magnets_ring_radius=2.2,
    magnets_ring_width=.2,
    # magnets_cylinder_num=0,
    magnets_cylinder_num=1,#1*plasma.nfp,
    magnets_cylinder_radius=2.1,
    magnets_cylinder_length=5,
    magnets_cylinder_phase=0,
    magnets_cylinder_thickness=0.05,
    supports_num=0,
    supports_width=.2,
  )
  reactor=freact.Reactor(plasma,params=params)

  log.info("generating component meshes")
  components=reactor.generate()
  for i,c in enumerate(components):
    log.info(f"component {i} vtx={len(c[0])} tri={len(c[1])}")
    
  # components=[clip_mesh(c) for c in components]

  path="reactor.html"
  log.info(f"generating visualization")
  viz_els=[
    el
    for idx,c in enumerate(components)
    for el in [
      viz.generate_mesh3d(c[0],c[1],color=rand.uniform(low=0,high=255,size=(3))),
      # viz.generate_mesh3d(c[0],c[1],color=[0,255,0],wireframe=True)
    ]
  ]
  log.info(f"writing visualization output={path}")
  viz.write_visualization(
    viz_els,
    f"{prefix}reactor.html"
  )
  
def test_reactor_torus():
  reactor_test_helper(ftplasma.TorusPlasma(10,4),"torus.")

def test_reactor_stellarator():
  reactor_test_helper(fsplasma.StellaratorPlasma(),"stellarator.")

if __name__=="__main__":
  test_reactor_torus()
  test_reactor_stellarator()
