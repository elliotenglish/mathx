from random_test_helpers import *
from mathx.robotics import jax_utilities
from mathx.robotics.ornstein_uhlenbeck_process import *
import scipy.stats
from mathx.core import log

def test_ornstein_uhlenbeck_process():
  log.initialize()

  mu=0.34
  sigma=.71
  theta=0
  proc=OrnsteinUhlenbeckProcess(mu=mu,sigma=sigma,theta=theta,
                                rand=jax_utilities.Generator(54324),
                                size=1)

  class Delta:
    def __init__(self):
      self.last=mu

    def __call__(self):
      x=proc()[0]
      #print(x)
      y=x-self.last
      self.last=x
      return y

  continuous_rng_test_helper(rng=Delta(),
                             pdf=lambda x:scipy.stats.norm.pdf(x,loc=0,scale=sigma),
                             low=-3*sigma,high=3*sigma)
