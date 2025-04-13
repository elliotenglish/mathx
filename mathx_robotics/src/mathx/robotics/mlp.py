import flax.nnx as nnx

class MLP(nnx.Module):
  def __init__(self,params):
    self.layers=[]
    features=params["in_features"]
    for lp in params["layers"]:
      self.layers.append(nnx.Linear(in_features=features,
                                    out_features=lp["out_features"],
                                    rngs=nnx.Rngs(43234)))
      if(lp["activation"]=="sigmoid"):
        self.layers.append(nnx.sigmoid)
      if(lp["activation"]=="relu"):
        self.layers.append(nnx.relu)
      features=lp["out_features"]

  def __call__(self,x):
    data=x
    for layer in self.layers:
      data=layer(data)
    return data
