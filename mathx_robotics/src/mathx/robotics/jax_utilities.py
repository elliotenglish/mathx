import jax
import jax.numpy as jnp

def in_jax(x):
  return isinstance(x,jax.core.Tracer)

def get_params(tree):
  pass

def get_shape(*args):
  return jax.numpy.broadcast_shapes(
    *[() if (a is None or isinstance(a,(float,int))) else a.shape for a in args])

class Generator:
  """
  Wrap JAX random key and generation functionality into a numpy compatible class
  """
  
  def __init__(self,seed):
    self.key=jax.random.key(seed)
    
  def increment(self):
    self.key,=jax.random.split(self.key,1)
    
  def uniform(self, low=0.0, high=1.0, size=None):
    x=jax.random.uniform(self.key,
                         shape=get_shape(low,high,size),
                         minval=low,
                         maxval=high)
    self.increment()
    return x
    
  def normal(self, loc=0.0, scale=1.0, size=None):
    x=jax.random.normal(self.key,
                        shape=get_shape(loc,scale,size))
    x=x*scale+loc
    self.increment()
    return x
    
  def integers(self, low, high=None, size=None, dtype=jnp.int32, endpoint=False):
    low=low if high is not None else 0
    high=high if high is not None else low
    x=jax.random.randint(self.key,
                         shape=get_shape(low,high,size),
                         dtype=dtype,
                         minval=low,
                         maxval=(high+1) if endpoint else high)
    self.increment()
    return x
