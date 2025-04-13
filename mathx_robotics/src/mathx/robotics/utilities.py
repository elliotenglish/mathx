import numpy as np
import jax.numpy as jnp
import jax.nn
import math

def one_hot_vector(n,i):
  # x=jnp.zeros(n)
  # x[i]=1
  # return x
  return jax.nn.one_hot(i,n)

def modulus_zero(num,denom):
  return num-math.floor(num/denom)*denom

def relative_angle(x0,x1):
  """Angle from x0 to x1"""
  x0=modulus_zero(x0,2*math.pi)
  x1=modulus_zero(x1,2*math.pi)
  if(x1>(x0+math.pi)):
    return x0+2*math.pi-x1
  if(x0>(x1+math.pi)):
    return x1+2*math.pi-x0
  return x1-x0
