from mathx.fusion import fuel_cycles
from mathx.core import log
import jax.numpy as jnp
import jax
import plotly.graph_objects as go

def test_fuel_cycles():
  Ts=jnp.linspace(0,1e10,100)
  cs={}
  for f in fuel_cycles.fuel_cycles:
    fuel=fuel_cycles.Fuel(f)
    cs[f]=jax.vmap(fuel.cross_section)(Ts)

  fig = go.Figure(data=[
    go.Scatter(
      name=k,
      x=Ts,
      y=v,
      mode='lines')
    for k,v in cs.items()
    ])
  fig.write_image("cross_sections.png")
