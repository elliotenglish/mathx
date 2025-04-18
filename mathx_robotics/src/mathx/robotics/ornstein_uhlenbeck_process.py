import jax.numpy as np

class OrnsteinUhlenbeckProcess:
  def __init__(self,mu,sigma,theta,size):
    self.mu=mu
    self.sigma=sigma
    self.theta=theta
    self.size=size
    
    self.reset()
    
  def reset(self):
    self.x=np.ones(self.size)*self.mu

  def sample(self,rand):
    x0=self.x
    self.x=self.theta*(self.mu-self.x)+rand.normal(scale=self.sigma)
    return x0
