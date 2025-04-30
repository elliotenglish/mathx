import numpy as np
import math
from dataclasses import dataclass

from . import curvilinear

class Box:
  @dataclass
  class Parameters:
    length: list[float]

  def __init__(self,
               params: Parameters = None,
               **kwargs):
    if params is not None:
      self.params=params
    else:
      self.params=Box.Parameters(**kwargs)

  def generate_density(self,num):
    l=np.array(self.params.length,dtype=np.float32)
    max_l=np.max(l)
    count=np.ceil(num*l/max_l).astype(np.int32)
    return count
  
  def uvw_to_xyz(self,u):
    return u*np.array(self.params.length,dtype=np.float32)

  def tesselate_volume(self,density):
    segments=self.generate_density(density)
    return curvilinear.generate_mesh_volume(
      lambda u:self.uvw_to_xyz(u),
      segments,
      closed=(False,False,False),
      degenerate=(None,None,None))

  def tesselate_surface(self,density):
    segments=self.generate_density(density)
    return curvilinear.generate_mesh_surface(
      lambda u:self.uvw_to_xyz(u),
      segments,
      closed=(False,False,False),
      degenerate=(None,None,None))
