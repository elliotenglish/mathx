from . import equilibrium
from .toroidal_plasma import ToroidalPlasma

class StellaratorPlasma(ToroidalPlasma):
  def __init__(self,eq=equilibrium.get_test_equilibrium()):
    self.eq=eq

  def get_surface(self,u):
    x,B=equilibrium.get_xyz_basis(self.eq,u)
    return x,B
  
  def get_B(self,u):
    return equilibrium.get_B(self.eq,u)

  def get_u(self,x):
    return equilibrium.get_u(self.eq,x)

  @property
  def nfp(self):
    return self.eq.NFP
