import jax.numpy as jnp

def orthogonalize(basis,normalize=True):
  basis_vecs=[]
  for v in basis:
    for bv in basis_vecs:
      v=v-(v@bv)*bv/jnp.linalg.norm(bv)**2
    if normalize:
      v=v/jnp.linalg.norm(v)
    basis_vecs.append(v)
  return jnp.array(basis_vecs)
