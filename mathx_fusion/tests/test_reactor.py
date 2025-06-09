import mathx.fusion.reactor as freact
import mathx.fusion.torus_plasma as ftplasma
import mathx.fusion.stellarator_plasma as fsplasma
import mathx.geometry.visualization as viz
from mathx.core import log
from mathx.geometry.mesh import Mesh

import numpy as np

def clip_mesh(mesh,x_plane):
  verts=[]
  vmap=dict()
  for i,v in enumerate(mesh.vertex):
    if v[0]<x_plane:
      vmap[i]=len(verts)
      verts.append(v)

  tris=[[vmap[int(vi)] for vi in t]
        for t in mesh.element
        if all([int(vi) in vmap for vi in t])]
  return Mesh(np.array(verts),np.array(tris))

colors={
  # "Plasma":(255,100,0),
  "Plasma":(255,0,0),
  # "PlasmaChamber":(180,195,205),
  "PlasmaChamber":(150,150,150),
  # "ConformalMagnet":(70,70,150),
  "ConformalMagnet":(0,100,200),
  # "RingMagnet":(180,120,80),
  "RingMagnet":(200,200,0),
  # "CylinderMagnet":(70,70,150),
  "CylinderMagnet":(50,180,50),
  "Support":(150,160,170),
  # "Port":(200,210,220),
  # "Port":(100,100,255),
  "Port":(0,200,50),
}

def reactor_test_helper(plasma,
                        params,
                        prefix="",
                        clip=False):
  rand=np.random.default_rng(54323)
  
  log.info("creating reactor")

  params={
    "wall_thickness":.15,
    "magnets_conformal_num":0,
    # "magnets_conformal_num":7*plasma.nfp,
    "magnets_conformal_width":.2,
    "magnets_ring_num":0,
    # "magnets_ring_num":3*plasma.nfp,
    "magnets_ring_radius":.5,
    "magnets_ring_width":.2,
    "magnets_cylinder_num":0,
    # "magnets_cylinder_num":2*plasma.nfp,
    "magnets_cylinder_radius":.8,
    "magnets_cylinder_length":4.5,
    "magnets_cylinder_phase":0,
    "magnets_cylinder_thickness":0.05,
    "ports_num":0,
    "ports_length":.5,
    "ports_radius":.005,
    "ports_thickness":.001,
    "supports_num":0,
    "supports_ground_level":-3,
    "supports_width":.01,
    **params
  }

  params=freact.ReactorParameters(**params)
  reactor=freact.Reactor(plasma,params=params)

  log.info("generating component meshes")
  components=reactor.generate()

  path="reactor.html"
  log.info(f"generating visualization")
  viz_els=[]
  density=64
  for c in components:
    mesh=c.object.tesselate_surface(density)
    if clip and c.material!="Plasma":
      mesh=clip_mesh(mesh,x_plane=7)
      # Prune empty components
      if len(mesh.element)==0:
        continue

    viz_els+=[
      viz.generate_mesh3d(mesh=mesh,color=colors[c.material]),
      #rand.uniform(low=0,high=255,size=(3))),
      # viz.generate_mesh3d(mesh=mesh,color=[0,255,0],wireframe=True)
    ]

  log.info(f"writing visualization output={path}")
  viz.write_visualization(
    viz_els,
    f"{prefix}reactor.html"
  )
  
def test_reactor_torus():
  reactor_test_helper(ftplasma.TorusPlasma(10,4),
                      params={},
                      prefix="Torus.")

def test_reactor_stellarator_conformal():
  plasma=fsplasma.StellaratorPlasma()
  reactor_test_helper(plasma,
                      params={
                        "magnets_conformal_num":7*plasma.nfp,
                      },
                      prefix="Stellarator.Conformal.")

def test_reactor_stellarator_cylinder():
  plasma=fsplasma.StellaratorPlasma()
  reactor_test_helper(plasma,
                      params={
                        "magnets_cylinder_num":7*plasma.nfp,
                      },
                      prefix="Stellarator.Cylinder.")
  
def test_reactor_stellarator_twin():
  plasma=fsplasma.StellaratorPlasma()
  reactor_test_helper(plasma,
                      params={
                        "magnets_conformal_num":7*plasma.nfp,
                        "magnets_ring_num":3*plasma.nfp,
                        "supports_num":12,
                        "ports_num":20
                      },
                      prefix="Stellarator.Twin.",
                      clip=True)

if __name__=="__main__":
  test_reactor_torus()
  test_reactor_stellarator_conformal()
  test_reactor_stellarator_cylinder()
