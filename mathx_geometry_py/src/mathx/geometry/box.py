import numpy as np
import math
from dataclasses import dataclass

from .curvilinear import Curvilinear

@dataclass
class Parameters:
  length: list[float]
  
def Box(params: Parameters = None, **kwargs):
  if params is None:
    params=Parameters(**kwargs)
  return Curvilinear(
    lambda u:u*np.array(params.length,dtype=np.float32))
