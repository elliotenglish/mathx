import jax
from dataclasses import dataclass

@dataclass
class Mesh:
  vertex: jax.Array
  element: jax.Array
