"""
https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
"""

import jax.numpy as jnp
from jax.typing import ArrayLike

import scipy

kB=scipy.constants.value("Boltzmann constant")

def velocity(v: jnp.array,
             m: float,
             T: ArrayLike) -> ArrayLike:
  """
  params:
    v: 3D velocity
    m: particle mass
    T: temperature
  """
  return (m/(2*jnp.pi*kB*T))**(3./2.)*jnp.exp(-m*v.T@v/(2*kB*T))

# def speed(s: ArrayLike,
#           m: float,
#           T: ArrayLike),