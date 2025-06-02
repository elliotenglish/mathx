#!/usr/bin/env python3

from mathx.fusion import equilibrium as eqx
from mathx.fusion import stellarator_plasma as spx
from mathx.fusion import reactor as reactx
from mathx.geometry import visualization as viz
from mathx.core import log
import argparse

def visualize(eq,prefix="",open=False):
  import desc
  desc.plotting.plot_1d(eq,"p")[0].savefig(f"{prefix}p.png")
  desc.plotting.plot_1d(eq,"ni")[0].savefig(f"{prefix}ni.png")
  desc.plotting.plot_1d(eq,"iota")[0].savefig(f"{prefix}iota.png")

  comp=reactx.PlasmaSurface(eq,1)
  eq_path=f"{prefix}eq.html"
  viz.write_visualization(
    [
      viz.generate_mesh3d(
        mesh=comp.tesselate_surface(64),
        opacity=1)
    ],
    eq_path
  )

  if open:
    import os
    # os.system(f"open {eq_path}")    
    os.system(f"google-chrome --new-window {eq_path}")

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  args=parser.parse_args()

  log.info("Computing equilibrium")

  params={
    "major_radius":8,
    "minor_radius":1.5,
    "NFP":3,
    "max_mode":1
  }

  plasma=spx.StellaratorPlasma(eqx.generate_equilibrium(params))
  visualize(plasma,open=True)
