from mathx.geometry.curvilinear import surface_normal_transform
from mathx.geometry import visualization as viz
from mathx.geometry import grid
import jax.numpy as jnp
import jax
import numpy as np

def normals_test_helper(name,pos_fn,us,norm_gt=None):
  norm_fn=surface_normal_transform(pos_fn)

  print(f"{us}")
  print(us.shape)

  print(f"{pos_fn(us[0])=}")
  print(f"{norm_fn(us[0])=}")

  xs=jax.vmap(pos_fn)(us)
  ns=jax.vmap(norm_fn)(us)
  print(f"{xs=}")
  print(f"{ns=}")

  if norm_gt is not None:
    for n in ns:
      np.testing.assert_allclose(n,jnp.array(norm_gt),atol=1e-4,rtol=1)

  viz.write_visualization(
    [
      viz.generate_points3d(xs,color=[255,0,0]),
      viz.generate_vectors3d(xs,ns,color=[0,255,0])
    ],
    f"normals.{name}.html"
  )


def test_surface_normal_transform():
  def pos_fn(u):
    return jnp.array([u[0],u[1],u[0]-.5*u[1]])
  normals_test_helper("flat",pos_fn,grid.generate_uniform_grid((3,4)),[-2./3.,1./3.,2./3.])

  def pos_fn(u):
    u=(u-.5)
    return jnp.array([u[0],u[1],-jnp.linalg.norm(u)])
  normals_test_helper("hyperbolic",pos_fn,grid.generate_uniform_grid((10,10)))
