import numpy as np
import jax
import jax.numpy as jnp

def expand(v,D):
  return v if isinstance(v,(list,tuple)) else tuple([v]*D)

def generate_uniform_grid(shape,flatten=True,endpoint=True,upper=1):
  D=len(shape)
  endpoint=expand(endpoint,D)
  upper=expand(upper,D)
  pts=np.meshgrid(*[
    np.linspace(0,upper[i],
                shape[i],
                endpoint=endpoint[i])
    for i in range(D)
    ])
  pts=np.concatenate([p[...,None] for p in pts],axis=D)
  if flatten:
    pts=pts.reshape((np.prod(np.array(shape)),D))
  return pts

def cylindrical_to_cartesian(rpz):
  return jnp.array([rpz[0]*jnp.cos(rpz[1]),rpz[0]*jnp.sin(rpz[1]),rpz[2]])

def cartesian_to_cylindrical(x):
  return jnp.array([jnp.linalg.norm(x[:2]),jnp.atan2(x[1],x[0]),x[2]])

def convert_cylindrical_to_cartesian(rpz,u):
  basis=jax.jacrev(cylindrical_to_cartesian)(rpz)
  return basis @ u
