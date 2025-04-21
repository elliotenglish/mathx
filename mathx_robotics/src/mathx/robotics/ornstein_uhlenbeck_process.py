import jax.numpy as np

class OrnsteinUhlenbeckProcess:
  def __init__(self,mu,sigma,theta,size,rand):
    self.mu=mu
    self.sigma=sigma
    self.theta=theta
    self.size=size
    self.rand=rand

    self.reset()
    
  def reset(self):
    self.x=np.ones(self.size)*self.mu

  def __call__(self):
    #print(f"sigma={self.sigma}")

    x0=self.x
    self.x=self.x+self.theta*(self.mu-self.x)+self.rand.normal(scale=self.sigma)
    return x0
