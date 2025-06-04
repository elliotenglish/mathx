#!/usr/bin/env python3

from mathx.fusion import equilibrium as eqx
from mathx.fusion import stellarator_plasma as spx
from mathx.fusion import reactor as reactx
from mathx.geometry import visualization as viz
from mathx.core import log
import argparse
import itertools
import os
from tqdm import tqdm

def visualize(plasma,prefix="",open=False):
  if isinstance(plasma,spx.StellaratorPlasma):
    import desc
    desc.plotting.plot_1d(plasma.eq,"p")[0].savefig(f"{prefix}p.png")
    desc.plotting.plot_1d(plasma.eq,"ni")[0].savefig(f"{prefix}ni.png")
    desc.plotting.plot_1d(plasma.eq,"iota")[0].savefig(f"{prefix}iota.png")

  comp=reactx.PlasmaSurface(plasma,1)
  surf_path=f"{prefix}surface.html"
  fig=viz.generate_visualization(
    [
      viz.generate_mesh3d(
        mesh=comp.tesselate_surface(64),
        opacity=1)
    ],
  )
  fig.write_html(surf_path)
  fig.write_image(f"{prefix}surface.png",width=1920,height=1080)

  if open:
    import os
    # os.system(f"open {surf_path}")
    os.system(f"google-chrome --new-window {surf_path}")

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  args=parser.parse_args()

  log.info("Computing equilibrium")

  search_params={
    "NFP":[3,4,5],
    "iota_edge":[1.5,2],
    "magnetic_well_weight":[5,10,15,20]
  }
  default_params={
    "optimizer":"lsq-auglag",
    "maxiter":250,
    "major_radius":8,
    "minor_radius":1,
    # "NFP":5,
    "mode_max":6,
    "pressure_core":1.8e4,
    # "iota_edge":1.5,
    "force_balance_weight":1e-5,#1e6,
    "quasisymmetry_two_term_weight":10,
    "quasisymmetry_triple_product_weight":0,
    # "magnetic_well_weight":10
  }

  keys=[x for x in search_params.keys()]
  values=[x for x in search_params.values()]
  log.info(f"{keys=}")
  log.info(f"{values=}")

  for perm_values in tqdm(itertools.product(*values)):
    params={**default_params,**dict(zip(keys,perm_values))}
    # log.info(f"{params=}")
    #log.info(perm_values)
    params=eqx.EquilibriumParameters(**params)
    #log.info(f"{params=}")

    prefix="output/"+"_".join([f"{k}_{v}" for k,v in zip(keys,perm_values)])+"/"
    if not os.path.exists(prefix):
      os.makedirs(prefix)
      plasma=spx.StellaratorPlasma(eqx.generate_equilibrium(params))
      visualize(plasma,
                prefix=prefix,
                open=False)
