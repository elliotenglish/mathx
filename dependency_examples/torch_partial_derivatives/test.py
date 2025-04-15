#!/usr/bin/env python3

import torch

x=torch.tensor([1],requires_grad=True,dtype=torch.float32)
y=torch.tensor([2],requires_grad=True,dtype=torch.float32)

z=(x**2+y**3).sum()
a=(x**3+y**2).sum()

print("x",x)
print("y",y)
print("z",z)
print("a",a)

print("x.grad",x.grad)
print("y.grad",y.grad)

#Need create_graph in order for dzdx to have a grad_fn value
dzdx=torch.autograd.grad(z,x,create_graph=True)[0]

print("x.grad",x.grad)
print("y.grad",y.grad)
print("dzdx",dzdx)

dzdx.backward(retain_graph=True)
print("x.grad",x.grad)
print("y.grad",y.grad)

d2zdx2=torch.autograd.grad(dzdx,x,create_graph=True)[0]
print("d2zdx2",d2zdx2)



while True:
  
