#!/usr/bin/env python3

import jax.numpy as jnp
import jax
from mathx.geometry import grid as gridx
from mathx.geometry import visualization as viz
from mathx.core import log
from mathx.core import jax_utilities
from mathx.fusion.torus_plasma import TorusPlasma
from mathx.fusion.stellarator_plasma import StellaratorPlasma
from mathx.fusion import reactor as freact

def compute_bounds(x):
  return jnp.min(x,axis=0),jnp.max(x,axis=0)

def compute_characteristic_length(x):
  low,high=compute_bounds(x)
  return jnp.max(high-low)

def compute_vector_length(v):
  return jnp.max(jnp.linalg.norm(v,axis=1))

def integrate_particle_through_em_field(field_fn,x0,v0,q,m,dt):
  """
  https://en.wikipedia.org/wiki/Particle-in-cell#The_particle_mover

  This implements the Boris algorithm (particle pusher) integration method for
  velocity and backward euler for position. Note that while the formal
  method definition has the velocity and position time staggered, you can still
  use the time collocated values with an O(dt) error and get the same
  integration stability properties.

  params:
    field_fn: F function with the signature `(position (in meters)) -> (B (in teslas), E (...))`
    x0: The initial position (at time t=0)
    v0: The initial velocity (at time t=0)
    q: The charge
    m: The mass (in kilograms)
    dt: The timestep
  returns:
    x1: The updated position (at time t=dt)
    v1: The updated velocity (at time t=dt)
  """
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
  integrate_fn=jax.vmap(integrate_particle_through_em_field,
                        in_axes=(None,0,0,0,0,None),
                        out_axes=(0,0))
  integrate_fn=jax.jit(integrate_fn,static_argnums=0)

  # Compute trajectories
  xt=x0.clone()
  vt=v0.clone()
  trajectories=[(x0,v0)]
  for i in range(num_steps):
    if i%10==0: log.info(f"step {i=} {xt[0]=} {vt[0]=}")
    xt,vt=integrate_fn(field_fn,xt,vt,q,m,dt)
    trajectories.append((xt,vt))

  return trajectories

def get_field_lines(field_fn,x0,dt,num_steps):
  def integrate_fn(x0):
    v=field_fn(x0)[0]
    v/=jnp.linalg.norm(v)
    x1=x0+dt*v
    return x1
  integrate_fn=jax.vmap(integrate_fn,in_axes=(0),out_axes=(0))
  integrate_fn=jax.jit(integrate_fn)

  # Compute trajectories
  xt=x0.clone()
  trajectories=[x0]
  for i in range(num_steps):
    if i%10==0: log.info(f"step {i=} {xt[0]=}")
    xt=integrate_fn(xt)
    trajectories.append(xt)

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
  log.info(f"{lower=} {upper=}")

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

def generate_particle_plasma_viz(name,plasma,num_particles,num_field_lines):
  rand=jax_utilities.Generator(543245)

  #setup field evaluation functions
  def field_fn_rtz(rtz):
    _,basis=plasma.get_surface(rtz)
    B=plasma.get_B(rtz)
    B=basis @ B
    return B,jnp.zeros(rtz.shape)

  def field_fn(x):
    rtz=plasma.get_u(x)
    return field_fn_rtz(rtz)

  field_fn_rtz_batch=jax.vmap(field_fn_rtz,in_axes=(0),out_axes=(0,0))
  field_fn_rtz_batch=jax.jit(field_fn_rtz_batch)
  
  surface_fn_batch=jax.vmap(plasma.get_surface,in_axes=(0),out_axes=(0,0))
  surface_fn_batch=jax.jit(surface_fn_batch)

  log.info("computing grid")
  grid=gridx.generate_uniform_grid((2,12,48),endpoint=(True,False,False),upper=[1,2*jnp.pi,2*jnp.pi])
  # grid=jnp.concatenate([grid,jnp.array([[1]]*grid.shape[0])],axis=1)
  # print(grid)

  log.info("computing EM field geometry")
  log.info("computing x,basis")
  surface_x,_=surface_fn_batch(grid)
  log.info("computing B")
  surface_B=field_fn_rtz_batch(grid)[0]

  log.info("computing magnetic field lines")
  field_x0,_=surface_fn_batch(
    rand.uniform(size=(num_field_lines,3))*jnp.array([[1,2*jnp.pi,2*jnp.pi]]))
  
  log.info(f"{field_x0=}")

  log.info("integrating")
  field_lines=get_field_lines(field_fn,field_x0,.01,2000)

  # log.info(f"{surface_x[:10]=}")
  # log.info(f"{surface_B[:10]=}")
  log.info(f"{field_lines[:3]}")

  max_vec_ratio=.03
  vec_scale=max_vec_ratio*compute_characteristic_length(surface_x)/compute_vector_length(surface_B)
  log.info(f"{vec_scale=}")

  plasma_component=freact.PlasmaSurface(plasma,1)

  viz.write_visualization(
    [
      viz.generate_mesh3d(mesh=plasma_component.tesselate_surface(64),
                          color=(255,0,0),opacity=0.2),
      viz.generate_lines3d(lines=[[field_lines[i][j] for i in range(len(field_lines))] for j in range(field_lines[0].shape[0])],
                           color=(0,255,0),
                           markers=False),
      viz.generate_vectors3d(surface_x.tolist(),(vec_scale*surface_B).tolist(),
      )#color=(0,255,255))
    ],
    f"{name}.magnetic_field.html"
  )

  log.info("compute particle trajectories")
  x0,_=surface_fn_batch(
    rand.uniform(size=(num_particles,3))*jnp.array([[1,2*jnp.pi,2*jnp.pi]]))
  v0=rand.uniform(size=(num_particles,3),low=-2.,high=2.)
  m=jnp.array([.02]*num_particles)
  q=jnp.array([1]*num_particles)

  log.info("integrating")
  particle_trajectories=get_trajectories(field_fn,x0,v0,m,q,.02,5000)

  viz.write_visualization(
    [
      viz.generate_mesh3d(mesh=plasma_component.tesselate_surface(64),
                          opacity=.2),
      viz.generate_lines3d([[d[0][i].tolist() for d in particle_trajectories] for i in range(x0.shape[0])],
                           (0,0,255))
    ],
    f"{name}.particle_trace.html")

if __name__=="__main__":
  # generate_particle_constant_B_viz()
  # generate_particle_cylindrical_B_viz()
  generate_particle_plasma_viz("torus",TorusPlasma(4,1.5),1,10)
  generate_particle_plasma_viz("stellarator",StellaratorPlasma(),1,10)
