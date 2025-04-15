import flax.nnx as nnx
import jax.numpy as jnp
import jax

class Mod(nnx.Module):
  def __init__(self,seed):
    self.lay0=nnx.Linear(in_features=4,out_features=3,rngs=nnx.Rngs(seed))
    self.lay1=nnx.Linear(in_features=3,out_features=2,rngs=nnx.Rngs(seed+1))

  def __call__(self,x):
    return self.lay1(self.lay0(x))

mod0=Mod(5432)
mod1=Mod(2345)

x=jnp.array([1,2,3,4])
print(f"x={x}")
print(f"y0={mod0(x)}")
print(f"y1={mod1(x)}")

_,state0=nnx.split(mod0)
_,state1=nnx.split(mod1)

# print(f"static={static}")
# print(f"state={state}")
# print(f"flatten={jax.tree.flatten(state)}")

# state["lay"]["bias"].value+=1

nnx.update(mod1,state0)
# print(f"state={state}")

print(f"y0={mod0(x)}")
print(f"y1={mod1(x)}")
