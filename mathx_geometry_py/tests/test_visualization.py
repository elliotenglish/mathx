import mathx.geometry.visualization as viz

def test_visualization():
  viz.write_visualization(
    [
      viz.generate_mesh3d(
        vtx=[[0,0,0],[1,0,0],[0,0,1]],
        tri=[[0,1,2]],
        color=(255,0,0)
      ),
      viz.generate_scatter3d(
        vtx=[[2,3,4]],
        color=(0,0,255)
      ),
      viz.generate_vectors3d(
        pts=[[-2,-1,2]],
        vecs=[[.2,.1,-.3]]
      )
    ],
    "viz.html")
