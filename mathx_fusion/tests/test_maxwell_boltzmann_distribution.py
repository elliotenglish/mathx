from mathx.fusion import maxwell_boltzmann_distribution as mbd
import functools
import jax.numpy as jnp
import numpy as np
import scipy
import mathx.core.log as log
import unittest

class TestMaxwellBoltzmannDistribution(unittest.TestCase):
  def test_velocity(self):
    Ts=[.65,2.43]
    lim=10

    for T in Ts:
      f=functools.partial(mbd.velocity,T=T,m=1.87)
      integral,integral_error=scipy.integrate.tplquad(lambda z,y,x:f(jnp.array([x,y,z])),-lim,lim,-lim,lim,-lim,lim)

      np.testing.assert_allclose(integral,1,atol=1e-3)

    # if mathx.core.testing.visualize

if __name__=="__main__":
  unittest.main()
