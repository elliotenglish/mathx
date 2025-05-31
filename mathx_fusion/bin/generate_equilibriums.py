#!/usr/bin/env python3

from mathx.fusion import equilibrium as eqx
from mathx.fusion import stellarator_plasma as spx
from mathx.fusion import reactor as reactx
from mathx.geometry import visualization as viz
from mathx.core import log
import argparse

def visualize(eq,path="eq.html",open=False):
  comp=reactx.PlasmaSurface(eq,1)
  viz.write_visualization(
    [
      viz.generate_mesh3d(
        mesh=comp.tesselate_surface(64),
        opacity=1)
    ],
    path
  )
  if open:
    import os
    # os.system(f"open {path}")    
    os.system(f"google-chrome --new-window {path}")

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
  visualize(plasma,path="eq.html",open=True)
