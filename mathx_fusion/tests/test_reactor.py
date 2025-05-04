import mathx.fusion.reactor as freact
import mathx.geometry.visualization as viz

def test_reactor():
  reactor=freact.Reactor(params=fusion.ReactorParams())

  vtx,tri_idx=reactor.generate()
  
  viz.write_visualization(
    [
      viz.generate_mesh3d(vtx,tri_idx,color=[255,0,0],wireframe=True)
    ],
    "reactor.html"
  )
