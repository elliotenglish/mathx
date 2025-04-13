import abc
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsps
import jax.nn as jnn

class Space:
  """
  A space describe the elements of a tensor.

  This is a reimplementation of:
  https://gymnasium.farama.org/api/spaces/fundamental/
  """

  @abc.abstractmethod
  def size(self):
    raise NotImplementedError

  # @abc.abstractmethod
  # def elements(self):
  #   raise NotImplementedError

  @abc.abstractmethod
  def real_size(self):
    raise NotImplementedError

  @abc.abstractmethod
  def to_real(self):
    raise NotImplementedError

  @abc.abstractmethod
  def to_action(self):
    raise NotImplementedError

  @abc.abstractmethod
  def apply_nonlinearity(self,linear):
    raise NotImplementedError

  @abc.abstractmethod
  def clamp(self,action):
    raise NotImplementedError

# def array_or_constant(x,i):
#   if isinstance(x,(int,float)):
#     return x
#   else
#     return x[i]

class ContinuousSpace(Space):
  def __init__(self,size,low=None,high=None,logarithmic=False):
    self.low=np.array([low]*size if isinstance(low,(int,float)) else low)
    self.high=np.array([high]*size if isinstance(high,(int,float)) else high)
    self.elements=[(float,float(self.low[i]),float(self.high[i]),logarithmic) for i in range(size)]

  def size(self):
    len(self.elements)

  # def element(self):
  #   return self.elements

  def real_size(self):
    return len(self.elements)

  def to_real(self,x):
    return x

  def from_real(self,x):
    return x

  def sample(self,rand,mean=None,stddev=None):
    if mean is not None:
      arr=self.clip(rand.normal(loc=mean,scale=stddev*.5*(self.high-self.low)))
    else:
      arr=rand.uniform(low=self.low,high=self.high,size=len(self.elements))
    return jnp.array(arr,dtype=jnp.float32)

  def clip(self,x):
    return jnp.clip(x,self.low,self.high)

  def apply_nonlinearity(self,linear):
    #We may want to use a sigmoid to avoid non-differentiability
    # return jnp.clip(linear,self.low,self.high)
    return jnn.sigmoid(linear)*(self.high-self.low)+self.low

  def normalize(self,x):
    return (x-self.low)/(self.high-self.low)*2-1

  def unnormalize(self,x):
    return (x+1)/2*(self.high-self.low)+self.low

class DiscreteSpace:
  def __init__(self,size,levels):
    self.elements=[(int,levels)]

  def size(self):
    len(self.elements)

  # def element(self):
  #   return self.elements

  def real_size(self):
    return sum([el[1] for el in self.elements])

  def to_real(self,x):
    arr=[]
    for el in self.elements:
      arr.extend([1 if x==i else 0 for i in range(el[1])])
    return np.array(arr,dtype=np.float32)

  def from_real(self,x):
    idx=0
    arr=[]
    for el in self.elements:
      arr.append(np.argmax(x[idx:idx+el[1]]))
      idx+=el[1]
    return np.array(arr)

  def sample(self,rand,mean=None,stddev=None):
    arr=[rand.integers(0,el[1]) for el in self.elements]
    return jnp.array(arr)

  def apply_nonlinearity(self,linear):
    idx=0
    nonlinear=[]
    for el in self.elements:
      nonlinear.append(jsps.softmax(linear[idx:idx+el[1]]))
      idx+=el[1]
    return jnp.concatenate(nonlinear)

# def space_to_real(space,x):
#   idx=0
#   arr=[]
#   for el in space.elements():
#     if el[0]==float:
#       arr.append(x[idx])
#       idx+=1
#     if el[0]==int:
#       for i in range(el[1]):
#         arr.append(1 if x[idx]==i else 0)
#       idx+=el[1]
#   return np.array(arr)

# def map_to_space_values(space,x):
#   idx=0
#   arr=[]
#   for el in space.elements():
#     if el[0]==float:
#       arr.append(x[idx])
#       idx+=1
#     if el[0]==int:
#       arr.append(np.argmax(x[idx:idx+el[1]]))
#       idx+=el[1]
#   return np.array(arr)

# def sample_action(space,rand):
#   arr=[]
#   for el in space.elements():
#     if el[0]==float:
#       arr.append(rand.uniform(el[1],el[2]))
#     if el[0]==int:
#       arr.append(rand.randint(0,el[1]-1))
