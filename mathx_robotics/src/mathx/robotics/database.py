import numpy as np
from mathx.core import log
import pickle

class Database:
  def __init__(self,params):
    self.data=[]
    self.max_size=params.get("max_size",None)
    self.load_factor=params.get("load_factor",0.9)

    log.info(f"Database max_size={self.max_size} load_Factor={self.load_factor}")

  def add(self,obs):
    self.data.append(obs)

    if self.max_size is not None and len(self.data)>self.max_size:
      l=len(self.data)
      log.info(f"pruning database old_size={l} new_size={len(self.data)}")
      self.data=self.data[int(l*(1-self.load_factor)):]

  def sample(self,num,rand):
    batch=[]
    for i in range(num):
      batch.append(self.data[rand.integers(low=0,high=len(self.data))])
    return batch

  def size(self):
    return len(self.data)

  # def to_array(self):
  #   if len(self.data)==0:
  #     return None#np.ndarray(3,0)
  #   return np.array(self.data)

  # def from_array(self,array):
  #   if array is not None:
  #     self.data=[x for x in array]
  #   self.data=[]

  def get(self):
    return self.data

  def print_stats(self):
    feedback=np.concatenate([p[3][None,...] for p in self.data])
    feedback=np.concatenate([p[3][None,...] for p in self.data])
    log.info(f"data={self.size()} feedback=[{feedback.min()},{feedback.max()},{np.mean(feedback)},{np.std(feedback)}] ")

  def save(self,path):
    with open(path,"wb") as f:
      pickle.dump(self.data,f)

  def restore(self,path):
    with open(path,"rb") as f:
      self.data=pickle.load(f)
