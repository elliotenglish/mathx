import mathx.fusion.reactor as freact
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

def test_reactor():
  log.info("creating reactor")
  reactor=freact.Reactor(params=freact.ReactorParameters())

  log.info("generating component meshes")
  components=reactor.generate()
  for i,c in enumerate(components):
    log.info(f"component {i} vtx={len(c[0])} tri={len(c[1])}")
    
  components=[clip_mesh(c) for c in components]

  path="reactor.html"
  log.info(f"generating visualization")
  viz_els=[
    el
    for idx,c in enumerate(components)
    for el in [
      viz.generate_mesh3d(c[0],c[1],color=np.random.uniform(low=0,high=255,size=(3)),wireframe=False),
      # viz.generate_mesh3d(c[0],c[1],color=[0,255,0],wireframe=True)
    ]
  ]
  log.info(f"writing visualization output={path}")
  viz.write_visualization(
    viz_els,
    "reactor.html"
  )

if __name__=="__main__":
  test_reactor()
