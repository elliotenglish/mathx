#!/usr/bin/env python3

from mathx.fusion import equilibrium as eqx
from mathx.fusion import stellarator_plasma as spx
from mathx.fusion import reactor as reactx
from mathx.geometry import visualization as viz
from mathx.core import log
import argparse

def visualize(plasma,prefix="",open=False):
  if isinstance(plasma,spx.StellaratorPlasma):
    import desc
    desc.plotting.plot_1d(plasma.eq,"p")[0].savefig(f"{prefix}p.png")
    desc.plotting.plot_1d(plasma.eq,"ni")[0].savefig(f"{prefix}ni.png")
    desc.plotting.plot_1d(plasma.eq,"iota")[0].savefig(f"{prefix}iota.png")

  comp=reactx.PlasmaSurface(plasma,1)
  surf_path=f"{prefix}surface.html"
  viz.write_visualization(
    [
      viz.generate_mesh3d(
        mesh=comp.tesselate_surface(64),
        opacity=1)
    ],
    surf_path
  )

  if open:
    import os
    # os.system(f"open {surf_path}")
    os.system(f"google-chrome --new-window {surf_path}")

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  args=parser.parse_args()

  log.info("Computing equilibrium")

  params={
    "major_radius":8,
    "minor_radius":1,
    "NFP":5,
    "max_mode":4
  }

  plasma=spx.StellaratorPlasma(eqx.generate_equilibrium(params))
  visualize(plasma,open=True)
