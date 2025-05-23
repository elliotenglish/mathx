import mathx.fusion.equilibrium as eqx
import jax
import jax.numpy as jnp
import desc

eq=eqx.get_test_equilibrium()

@jax.custom_batching.custom_vmap
def compute(rtz):
  grid=desc.grid.Grid(nodes=rtz[None,...],jitable=True)
  out=eq.compute(["X"],grid)
  return out["X"][0]

@compute.def_vmap
def compute_vmap(axis_size,in_batched,rtzs):
  # print(f"{axis_size=} {in_batched=} {rtzs=}")
  # grid=desc.grid.Grid(nodes=rtzs,jitable=True)
  # out=eq.compute(["X"],grid)
  # return out["X"]
  return rtzs,True

compute_batch=jax.vmap(compute,in_axes=(0),out_axes=(0))
compute_batch=jax.jit(compute_batch)

in_=jnp.array([[0.5,0,0],[.25,.25,.25]])
out=compute_batch(in_)

print(in_)
print(out)
