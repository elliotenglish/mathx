import mathx.fusion.reactor as freact
import mathx.geometry.visualization as viz
from mathx.core import log

def test_reactor():
  log.info("creating reactor")
  reactor=freact.Reactor(params=freact.ReactorParameters())

  log.info("generating component meshes")
  components=reactor.generate()

  path="reactor.html"
  log.info(f"visualizing output={path}")
  viz.write_visualization(
    [
      el
      for c in components
      for el in [
        viz.generate_mesh3d(c[0],c[1],color=[0,0,255],wireframe=False),
        viz.generate_mesh3d(c[0],c[1],color=[0,255,0],wireframe=True)
      ]
    ],
    "reactor.html"
  )

if __name__=="__main__":
  test_reactor()
