import mathx.fusion.equilibrium as eqx
import mathx.core.log as log
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jopt
import desc

def test_compute():
  eq=eqx.get_test_equilibrium()

  @jax.custom_batching.custom_vmap
  def compute(rtz):
    grid=desc.grid.Grid(nodes=rtz[None,...],jitable=True)
    out=eq.compute(["X"],grid)
    return out["X"][0]

  @compute.def_vmap
  def compute_vmap(axis_size,in_batched,rtzs):
    # print(f"{axis_size=} {in_batched=} {rtzs=}")
    grid=desc.grid.Grid(nodes=rtzs,jitable=True)
    out=eq.compute(["X"],grid)
    return out["X"]
    return rtzs,True

  compute_batch=jax.vmap(compute,in_axes=(0),out_axes=(0))
  compute_batch=jax.jit(compute_batch)

  in_=jnp.array([[0.5,0,0],[.25,.25,.25]])
  out=compute_batch(in_)

  log.info(f"{in_=}")
  log.info(f"{out=}")

def test_map_coordinates():
  eq=eqx.get_test_equilibrium()
  
  rtz0=jnp.array([[.5,.5*jnp.pi,.25*jnp.pi],[.3,.25*jnp.pi,.1*jnp.pi]])
  num_nodes=rtz0.shape[0]
  
  def compute_xyz_fn(rtz):
    x=eq.compute(["X","Y","Z"],desc.grid.Grid(nodes=rtz,jitable=True))
    x=jnp.concatenate([x["X"][...,None],
                       x["Y"][...,None],
                       x["Z"][...,None]],
                      axis=1)
    return x
  x0=compute_xyz_fn(rtz0)

  rtz_init=jnp.concatenate([jnp.zeros((num_nodes,1))+.001,
                            jnp.zeros((num_nodes,1))+.001,
                            jnp.arctan2(x0[:,1],x0[:,0])[...,None]],
                           axis=1)

  def obj_fn(rtz):
    rtz=rtz.reshape(-1,3)
    return jnp.linalg.norm(compute_xyz_fn(rtz)-x0)**2

  log.info(f"{obj_fn(rtz0)=}")
  log.info(f"{obj_fn(rtz_init)=}")

  # obj_grad_fn=jax.value_and_grad(obj_fn)
  # obj_grad_fn=jax.jit(obj_grad_fn)
  # step_size=.005
  # rtz_k=rtz_init
  # for j in range(10000):
  #   obj,grad=obj_grad_fn(rtz_k)
  #   rtz_k=rtz_k-step_size*grad
  #   print(f"{obj=} {rtz_k=}")
  # print(f"{obj_fn(rtz_k,x0)}=")

  minimize=jax.jit(jopt.minimize,static_argnums=(0),static_argnames=("method"))
  for j in range(10):
    rtz_k=rtz_init
    rtz_k=minimize(obj_fn,rtz_k.reshape(-1),method="BFGS").x.reshape(-1,3)
    log.info(f"{obj_fn(rtz_k)}=")

  # rtz=eq.map_coordinates(x,inbasis=("X","Y","Z"),outbasis=("rho","theta","zeta"))

if __name__=="__main__":
  #test_compute()
  test_map_coordinates()