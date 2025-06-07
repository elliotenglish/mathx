import itertools
import jax.numpy as jnp

class FourierND:
  def __init__(self,mode_shape=None,modes=None,coefficients=None):
    assert (mode_shape is not None) != (modes is not None)

    if modes:
      self.modes=modes.copy()
    else:
      ranges=[range(-(m//2),m//2+1) for m in mode_shape]
      self.modes=[m for m in itertools.product(*ranges)]

    if coefficients is not None:
      self.coefficients=coefficients.copy()
    else:
      self.coefficients=jnp.zeros([len(self.modes)])

  def __call__(self,x):
    value=0
    for i,m in enumerate(self.modes):
      basis=1
      for j,d in enumerate(m):
        a=2*jnp.pi*m[j]*x[j]
        if m[j]<=0:
          basis*=jnp.cos(-a)
        else:
          basis*=jnp.sin(a)
      value+=self.coefficients[i]*basis
    return value
