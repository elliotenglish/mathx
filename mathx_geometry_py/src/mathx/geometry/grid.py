import numpy as np

def generate_uniform_grid(shape,flatten=True):
  D=len(shape)
  pts=np.meshgrid(*[np.linspace(0,1,s) for s in shape])
  pts=np.concatenate([p[...,None] for p in pts],axis=D)
  if flatten:
    pts=pts.reshape((np.prod(np.array(shape)),D))
  return pts
