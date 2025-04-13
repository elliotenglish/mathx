import abc

class System:
  @abc.abstractmethod
  def state_space(self):
    raise NotImplementedError
  
  @abc.abstractmethod
  def action_space(self):
    raise NotImplementedError

  @abc.abstractmethod
  def state(self):
    raise NotImplementedError

  @abc.abstractmethod
  def transition(self,action):
    raise NotImplementedError
  
  @abc.abstractmethod
  def feedback(self):
    raise NotImplementedError

  @abc.abstractmethod
  def done(self):
    return False

  @abc.abstractmethod
  def reset(self,rand):
    NotImplementedError
