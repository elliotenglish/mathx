from mathx.geometry.curvilinear import surface_normal_transform
from mathx.geometry import visualization as viz
import jax.numpy as jnp
import jax
import numpy as np

def generate_uniform_grid(shape,flatten=True):
  D=len(shape)
  pts=jnp.meshgrid(*[jnp.linspace(0,1,s) for s in shape])
  pts=jnp.concat([p[...,None] for p in pts],axis=D)
  if flatten:
    pts=pts.reshape((jnp.prod(jnp.array(shape)),D))
  return pts

def test_surface_normal_transform():
  def pos_fn(u):
    return jnp.array([u[0],u[1],u[0]-.5*u[1]])#-jnp.linalg.norm(u))
  norm_fn=surface_normal_transform(pos_fn)

  us=generate_uniform_grid((3,4))
  print(f"{us}")
  print(us.shape)

  print(f"{pos_fn(us[0])=}")
  print(f"{norm_fn(us[0])=}")

  xs=jax.vmap(pos_fn)(us)
  ns=jax.vmap(norm_fn)(us)
  print(f"{xs=}")
  print(f"{ns=}")
  
  for n in ns:
    np.testing.assert_allclose(n,jnp.array([-2./3.,1./3.,2./3.]),atol=1e-4,rtol=1)

  viz.write_visualization(
    [
      viz.generate_points3d(xs,color=[255,0,0]),
      viz.generate_vectors3d(xs,ns,color=[0,255,0])
    ],
    "normals.html"
  )
