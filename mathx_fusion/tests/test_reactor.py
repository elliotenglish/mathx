import mathx.fusion.reactor as freact
import mathx.geometry.visualization as viz

def test_reactor():
  reactor=freact.Reactor(params=freact.ReactorParameters())

  vtx,tri_idx=reactor.generate()
  
  viz.write_visualization(
    [
      viz.generate_mesh3d(vtx,tri_idx,color=[0,0,255],wireframe=False),
      viz.generate_mesh3d(vtx,tri_idx,color=[0,255,0],wireframe=True)
    ],
    "reactor.html"
  )

if __name__=="__main__":
  test_reactor()
