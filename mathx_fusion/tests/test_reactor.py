import mathx.fusion.reactor as freact
import mathx.geometry.visualization as viz
from mathx.core import log

def test_reactor():
  log.info("creating reactor")
  reactor=freact.Reactor(params=freact.ReactorParameters())

  log.info("generating component meshes")
  components=reactor.generate()
  for i,c in enumerate(components):
    log.info(f"component {i} vtx={len(c[0])} tri={len(c[1])}")

  path="reactor.html"
  log.info(f"generating visualization")
  viz_els=[
    el
    for c in components
    for el in [
      viz.generate_mesh3d(c[0],c[1],color=[0,0,255],wireframe=False),
      viz.generate_mesh3d(c[0],c[1],color=[0,255,0],wireframe=True)
    ]
  ]
  log.info(f"writing visualization output={path}")
  viz.write_visualization(
    viz_els,
    "reactor.html"
  )

if __name__=="__main__":
  test_reactor()
