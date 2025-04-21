from mathx.robotics import jax_utilities
import math
import scipy.stats

from random_test_helpers import *

def test_jax_utilities_generator():
  generator=jax_utilities.Generator(54324)
  
  continuous_rng_test_helper(rng=lambda:generator.uniform(low=-1.5,high=.75),
                             pdf=lambda x:1./(.75-(-1.5)),
                             low=-1.5,high=.75)

  continuous_rng_test_helper(rng=lambda:generator.normal(loc=1.2,scale=2.3),
                             pdf=lambda x:scipy.stats.norm.pdf(x,loc=1.2,scale=2.3),
                             low=1.2-3*2.3,high=1.2+3*2.3)

  discrete_rng_test_helper(rng=lambda:generator.integers(low=0,high=6),
                           p=lambda x:float(1)/6,
                           num=6)
