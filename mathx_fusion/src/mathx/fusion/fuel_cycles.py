import jax.numpy as jnp
import jax
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

def keV(T):
  return T/ev_in_K*1e-3

class Fuel:
  def __init__(self,name):
    self.fuel_data=[d for d in get_data() if d["name"]==name][0]

  def cross_section(self,T):
    if "duane_coefficients" in self.fuel_data:
      E=keV(T)
      return duane_function(E,self.fuel_data["duane_coefficients"])
    raise ValueError
  
  def reactivity(self,T):
    """
    For a given T this is the Maxwellian average velocity-cross section product.
    """

    Ts=(0,2e9,100)
    
    Es=keV(T)
    ss=jax.vmap(self.cross_section)(Es)
    
    dT=Ts[1]-Ts[0]

    mu=m1*m2/(m1+m2)

    reactivity=4/(2*jnp.pi*m1)**.5 * (mu/(m1*k*T))**1.5 * jnp.sum(Es*ss*jnp.exp(mu*Es/(m1*k*T)*dT)
                                                                  
                                                                  return reactivity

  # def velocity(self,T,rho):
