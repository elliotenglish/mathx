#!/usr/bin/env python3

import torch
import argparse
import time

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("--device",required=True,help="cpu, cuda")
  parser.add_argument("--format",required=True,help="nhwc, nchw")
  parser.add_argument("--channels",type=int,default=128)
  parser.add_argument("--kernel",type=int,default=3)
  parser.add_argument("--resolution",type=int,default=128)
  parser.add_argument("--iterations",type=int,default=1000)
  args=parser.parse_args()

  if args.format=="nhwc":
    memory_format=torch.channels_last
  elif args.format=="nchw":
    memory_format=torch.contiguous_format
  else:
    assert False
    
  device=torch.device(args.device,0)

  x=torch.zeros([1,args.channels,1,args.resolution],dtype=torch.float32).to(memory_format=memory_format,device=device)
  weights=torch.zeros([args.channels,args.channels,1,args.kernel],dtype=torch.float32).to(memory_format=memory_format,device=device)
  print(f"x shape={x.shape} dim_order={x.dim_order()}")
  print(f"weights shape={weights.shape} dim_order={weights.dim_order()}")

  y=torch.nn.functional.conv2d(x,weights)
  t0=time.time()
  for it in range(args.iterations):
    y=torch.nn.functional.conv2d(x,weights)
  t1=time.time()

  print(f"y shape={y.shape} dim_order={y.dim_order()}")

  print(f"dt={t1-t0}")
