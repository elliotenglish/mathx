import acme
import abc

class Agent(acme.Actor):
  def __init__(self):
    pass

  @abc.abstractmethod
  def save(self,path):
    raise NotImplementedError

  @abc.abstractmethod
  def restore(self,path):
    raise NotImplementedError

  @abc.abstractmethod
  def q(self,x,a):
    raise NotImplementedError
