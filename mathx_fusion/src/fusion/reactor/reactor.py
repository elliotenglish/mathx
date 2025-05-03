import numpy as np
import paramak
import math

from mathx.geometry.torus import Torus
from mathx.geometry.cylinder import Cylinder
from .magnet import Magnet

from dataclasses import dataclass

@dataclass
class ReactorParameters:
  primary_chamber: Torus.Parameters
  magnets: list[Magnet.Parameters]
  # heating_ports: list[Port.Parameters]
  # diagnostic_ports: list[Port.Parameters]
  # diverter_ports: list[Port.Parameters]

def surface_fn(u):
  """
  params:
    u: \vec{u}=(u,v,w) as a np.ndarray, u \in [0,2*pi], v\in [0,2*pi], w in [0,inf]

  return:
    x: \vec{x}=(x,y,z) as a np.ndarray

  Algorithm:
    \vec{xs}=posfn(u,v,0)

    du=\frac{\partial\vec{xs}}{u}
    dv=\frac{\partial\vec{xs}}{u}
    \vec{n}=\frac{d0\times d1}{|d0||d1|}

    \vec{x}=\vec{xs}+w*\vec{n}
  """

class Reactor:
  def __init__(self, params: ReactorParameters):
    self.plasma_chamber=Torus(params.plasma_chamber)

  def Compute(self):
    #1 Define plasma chamber using surface_fn
    #2 Define magnets as grid on offset surface_fn
    #3 Define ports using surface_fn
    #4 Define support stucture by sampling extrema of surface function and then extruding to ground plane
    #4 Define exterior mechanisms...
