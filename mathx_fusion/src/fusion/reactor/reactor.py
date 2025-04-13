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

class Reactor:
  def __init__(self, params: ReactorParameters):
    self.plasma_chamber=Torus(params.plasma_chamber)
    self.magnets=
    
  def Compute(self):
    pass
