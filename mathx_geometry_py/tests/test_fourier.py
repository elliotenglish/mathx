from mathx.geometry.fourier import FourierND
from mathx.core import log
import jax.numpy as jnp

def test_fourier_nd():
  series=FourierND(mode_shape=(1,2))
  series.coefficients.at[0].set(1)
  series.coefficients.at[1].set(1)
  log.info(f"{series.modes=}")
  log.info(f"{series.coefficients=}")
  x=jnp.array([0.1,1.2])
  log.info(f"{x=}")
  log.info(f"{series(x)=}")
