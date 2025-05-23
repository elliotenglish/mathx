import jax
import jax.numpy as jnp

@jax.custom_batching.custom_vmap
def f_fwd(x, y):
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

@f_fwd.def_vmap
def f_fwd_vmap(_, in_batched, x, y):
  # Insert a new const here to test the optimize_remat batching rule.
  out = np.array([2.0])*f(x, y)
  out_batched = (True, (True, True, True))
  return (out, (jnp.cos(x), jnp.sin(x), y)), out_batched

jax.jit(jax.vmap(f_fwd,in_axes=(0,0),out_axes=(0,0)))
