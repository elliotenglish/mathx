import jax.numpy as jnp
import jax
from mathx.geometry import grid as gridx
from mathx.geometry import visualization as viz
from mathx.core import log
from mathx.fusion import equilibrium

def integrate_particle_through_em_field(field_fn,x0,v0,q,m,dt):
  # https://arxiv.org/abs/2410.03352v1
  if False:
    B,E=field_fn(x0)
    f=(q*(E+jnp.cross(v0,B)))
    v1=v0+dt*f/m
    x1=x0+dt*v1
  else:
    B,E=field_fn(x0)
    qp=dt*q/(2*m)
    h=qp*B
    s=2*h/(1+jnp.linalg.norm(h)**2)
    u=v0+qp*E
    up=u+jnp.cross(u+(jnp.cross(u,h)),s)
    v1=up+qp*E
    x1=x0+dt*v1
  return x1,v1

def get_trajectories(field_fn,x0,v0,m,q,dt,num_steps):
  xt=x0.clone()
  vt=v0.clone()
  
  integrate_fn=jax.vmap(integrate_particle_through_em_field,
                        in_axes=(None,0,0,0,0,None),
                        out_axes=(0,0))

  # Compute trajectories
  trajectories=[(x0,v0)]
  for i in range(num_steps):
    log.info(f"step {i=}")
    xt,vt=integrate_fn(field_fn,xt,vt,q,m,dt)
    trajectories.append((xt,vt))

  return trajectories

def generate_particle_viz(field_fn,x0,v0,m,q,path):
  dt=.01
  num_steps=1000

  trajectories=get_trajectories(field_fn,x0,v0,dt,num_steps)

  # Compute trajectory bounds
  lower=jnp.array([jnp.inf]*3)
  upper=-lower
  for d in trajectories:
    lower=jnp.minimum(lower,d[0].min(axis=0))
    upper=jnp.maximum(upper,d[0].max(axis=0))
  print(lower,upper)
  
  # Compute magnetic fields
  dul=upper-lower
  ranges=[jnp.linspace(lower[d],upper[d],
                       max(1,int(20*(dul[d])/max(dul)))) for d in range(3)]

  grid=jnp.meshgrid(*ranges)
  grid=jnp.concatenate([g[...,None] for g in grid],axis=3).reshape(-1,3)
  field=jax.vmap(lambda x:field_fn(x)[0]*.05,in_axes=(0),out_axes=(0))(grid)

  viz.write_visualization(
    [
      viz.generate_lines3d([[d[0][i].tolist() for d in trajectories] for i in range(x0.shape[0])],
                           (255,0,0)),
      # viz.generate_vectors3d(grid,field,
      #                        (0,255,0))
    ],
    path)

def generate_particle_constant_B_viz():
  def field_fn(x):
    return jnp.array([1,0,0]),jnp.array([0,0,0])

  x0=jnp.array([[0,1,0],[0,.5,.5]])
  v0=jnp.array([[1,0,-1],[1,.5,-.5]])
  m=jnp.array([1,.1])
  q=jnp.array([1,-1])
  
  generate_particle_viz(field_fn,x0,v0,m,q,"particle_constant_B.html")
  
def generate_particle_cylindrical_B_viz():
  def field_fn(x):
    return jnp.array([x[1],-x[0],0]),jnp.array([0,0,0])
  
  x0=jnp.array([[1,0,0]])
  v0=jnp.array([[0,1,-1]])
  m=jnp.array([.01])
  q=jnp.array([1])

  generate_particle_viz(field_fn,x0,v0,m,q,"particle_cylindrical_B.html") 

def generate_particle_equilibrium_viz():
  # u=jnp.array([[0,0,.1]]

  # dt=.01
  # num_steps=1000
  
  grid=gridx.generate_uniform_grid((64,16,2),endpoint=False)
  # grid=jnp.concatenate([grid,jnp.array([[1]]*grid.shape[0])],axis=1)
  # print(grid)
  
  eq=equilibrium.get_test_equilibrium()

  x=equilibrium.get_xyz_basis(eq,grid)[0]
  B=equilibrium.get_B(eq,grid)
  
  rtz=equilibrium.get_u(eq,x)
  
  def field_fn(x):
    def pure_callback(x):
      xs=x[None,:] if len(x.shape)==1 else x
      rtz=equilibrium.get_u(eq,xs)
      B=equilibrium.get_B(eq,rtz)
      B=B[0] if len(x.shape)==1 else B
      return B,jnp.zeros(B.shape)
    
    return jax.pure_callback(pure_callback,
                             (jax.ShapeDtypeStruct((3,),x.dtype),
                              jax.ShapeDtypeStruct((3,),x.dtype)),
                             x,
                             vmap_method="expand_dims")
    
  x0=equilibrium.get_xyz_basis(eq,jnp.array([[0,0,.5]]))[0]
  v0=jnp.array([[0.5,0.5,0.5]])
  m=jnp.array([.01])
  q=jnp.array([1])

  dt=.001
  num_steps=1000
  trajectories=get_trajectories(field_fn,x0,v0,m,q,dt,num_steps)

  # print(f"{grid[0]=}")
  # print(f"{rtz[0]=}")
  # print(f"{x[0]=}")

  viz.write_visualization(
    [
      viz.generate_lines3d([[d[0][i].tolist() for d in trajectories] for i in range(x0.shape[0])],
                           (255,0,0)),
      viz.generate_vectors3d(x,B*.1,
                             (0,255,0))
    ],
    "particle_equilibrium_B.html")

if __name__=="__main__":
  # generate_particle_constant_B_viz()
  # generate_particle_cylindrical_B_viz()
  generate_particle_equilibrium_viz()
