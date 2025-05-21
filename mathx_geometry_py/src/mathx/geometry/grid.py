import numpy as np
import jax
import jax.numpy as jnp

def generate_uniform_grid(shape,flatten=True,endpoint=True):
  D=len(shape)
  pts=np.meshgrid(*[np.linspace(0,1,s,endpoint=endpoint) for s in shape])
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
