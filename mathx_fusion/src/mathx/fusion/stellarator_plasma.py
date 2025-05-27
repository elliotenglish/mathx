from . import equilibrium
from .toroidal_plasma import ToroidalPlasma

def StellaratorPlasma(ToroidalPlasma):
  def __init__(self):
    self.eq=equilibrium.get_test_equilibrium()

  def get_surface(self,u):
    return equilibrium.get_xyz_basis(self.eq,u)
  
  def get_B(self,u):
    return equilibrium.get_B(self.eq,u)

  def get_u(self,x):
    return equilibrium.get_u(self.eq,x)
