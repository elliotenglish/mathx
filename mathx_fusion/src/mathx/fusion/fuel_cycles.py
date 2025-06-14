import jax.numpy as jnp
import json
import scipy

fuel_cycles=["DT","DHe3"]

def duane_function(E,A):
  """
  params:
    E: The energy in keV
    A: The Duane coefficients
  returns:
    The cross section in barns (1 barn = 10^-24 cm^2)
  """
  return (A[4]+A[1]/((A[3]-A[2]*E)**2+1))/(E*jnp.exp(A[0]*E**-.5)-1)

ev_in_K=scipy.constants.value("electron volt-kelvin relationship")

data=None
def get_data():
  global data
  if data is None:
    from importlib.resources import files
    with files("mathx.fusion").joinpath("fuel_cycles.json").open("r") as f:
      data=json.load(f)
  return data
    
class Fuel:
  def __init__(self,name):
    self.fuel_data=[d for d in get_data() if d["name"]==name][0]

  def cross_section(self,T):
    if "duane_coefficients" in self.fuel_data:
      E=T/ev_in_K*1e-3
      return duane_function(E,self.fuel_data["duane_coefficients"])
    raise ValueError

  # def velocity(self,T,rho):
