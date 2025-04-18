#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import json
import re
import math
import numpy as np
import datetime
import abc
import csv

float_pattern="[-+]?(?:[0-9]+[.]?[0-9]*|[0-9]*[.][0-9]+)(?:[eE][-+]?[0-9]+)?"
timestamp_pattern="[0-9]*-[0-9]*-[0-9]* [0-9]*:[0-9]*:[0-9]*[.,][0-9]*"
# timestamp_pattern="([0-9]*)-([0-9]*)-([0-9]*) ([0-9]*):([0-9]*):([0-9]*)[.,]([0-9]*)"

def split_log_line(line):
  """
  Splits a line of the format:
  [<TIMESTAMP>]<STRING>

  returns:
  (epoch_time(<TIMESTAMP>),<STRING>)
  """

  log_pattern=f"\\[({timestamp_pattern})\\] (.*)"
  match=re.match(log_pattern,line)
  if match is not None:
    ts=datetime.datetime.fromisoformat(match.group(1))
    return ts.timestamp(),match.group(2)
  log_pattern=f"\\[({float_pattern})\\] (.*)"
  match=re.match(log_pattern,line)
  if match is not None:
    return float(match.group(1)),match.group(2)

  raise ValueError

def parse_key_values(line):
  """
  Parses a line of the format:
  <KEY0>=<VALUE0> <KEY1>=<VALUE1> ...

  returns:
  {<KEY0>:<VALUE0>, <KEY1>:<VALUE1>, ...}
  """
  kv_pattern=f"([A-Za-z0-9_]*)=(\\[?(?:[ ]*{float_pattern})+[ ]*\\]?|[A-Za-z0-9.,_]*)"
  matches=re.findall(kv_pattern,line)

  # print(kv_pattern)
  # print(line)
  # print(matches)

  kv={}
  for m in matches:
    if "[" in m[1]:
      kv[m[0]]=[float(x) for x in m[1][1:-1].split(" ") if len(x)>=1]
    else:
      try:
        kv[m[0]]=float(m[1])
      except:
        kv[m[0]]=m[1]

    # if m[0]=="episode_feedback":
    #   print(m[0],m[1],type(m[1]),kv[m[0]])
    #   time.sleep(1)

  # print(line,kv)

  return kv

def add_line(data,timestamp,line_kv,xk,yk):
  for k in yk:
    if k not in data: data[k]=[]
    kp=k.split(".")
    if len(kp)>1:
      k0=kp[0]
      k1=int(kp[1]) if len(kp)>1 else 0
      if k0 in line_kv:
        data[k].append((timestamp,line_kv[k0][k1]))
    elif k in line_kv:
      data[k].append((timestamp,line_kv[k]))

class BaseLog:
  def __init__(self,path,xk,yk):
    self.f=open(path,"r")
    self.data={}
    self.xk=xk
    self.yk=yk

  def get_data(self):
    self.update()
    return self.data
  
  def update(self):
    for line in self.f:
      if line:
        try:
          line_timestamp,line_str=split_log_line(line)
        except:
          continue
        # print(line_timestamp)

        line_data=self.parse_line(line_str)
        if line_data:
          add_line(self.data,line_timestamp,line_data,self.xk,self.yk)

  @abc.abstractmethod
  def parse_line(self,line):
    raise NotImplementedError

class JSONLog(BaseLog):
  def parse_line(self,line):
    try:
      return json.loads(line)
    except json.JSONDecodeError as e:
      pass
    except:
      print(f"parsing line failed: {str}")
      raise
    return None

class KVLog(BaseLog):
  def parse_line(self,line):
    return parse_key_values(line)
  
class CSVLog(BaseLog):
  def __init__(self,*args):
    BaseLog.__init__(self,*args)
    
    self.reader=csv.reader(self.f)
    self.header=next(self.reader)
    self.idx=0

  def update(self):
    for row in self.reader:
      line_data={self.header[i]:float(row[i]) for i in range(len(self.header))}
      add_line(self.data,self.idx,line_data,self.xk,self.yk)
      self.idx+=1

def decimate_points(x,num):
  if len(x)<=num:
    return x

  bucket_size=math.floor(len(x)/num)
  bucket_remainder=len(x)%num
  # print(len(x),num,bucket_size,bucket_remainder)

  x_new=[]

  for i in range(num):
    if i<bucket_remainder:
      begin=i*(bucket_size+1)
      end=begin+(bucket_size+1)
    else:
      begin=bucket_remainder*(bucket_size+1)+(i-bucket_remainder)*bucket_size
      end=begin+bucket_size
    # print(i,begin,end)
    x_new.append((sum([p[0] for p in x[begin:end]])/(end-begin),
                  sum([p[1] for p in x[begin:end]])/(end-begin)))

  # print(x_new)

  return x_new

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("--files",nargs="*",required=True)
  parser.add_argument("--json",action="store_true")
  parser.add_argument("--kv",action="store_true")
  parser.add_argument("--csv",action="store_true")
  parser.add_argument("-x")#,required=True)
  parser.add_argument("-y",nargs="*",required=True)
  parser.add_argument("--decimation",type=int,default=100)
  parser.add_argument("--absolute_time",action="store_true")
  args=parser.parse_args()

  fig, axs = plt.subplots(len(args.y), 1, sharex=True, figsize=(8, 6), squeeze=False)

  logs={}
  for path in args.files:
    if args.json:
      logs[path]=JSONLog(path,args.x,args.y)
    elif args.kv:
      logs[path]=KVLog(path,args.x,args.y)
    elif args.csv:
      logs[path]=CSVLog(path,args.x,args.y)
    else:
      assert False

  for path in args.files:
    print(f"parsing {path}")
    data=logs[path].get_data()
    # print(data["episode_feedback"][:20])
    # print(data)

    if args.absolute_time:
      t0=0
    else:
      t0=min([data[k][0][0] for k in args.y])

    print(f"plotting")
    for idx,k in enumerate(args.y):
      data_offset=[(x[0]-t0,x[1]) for x in data[k]]
      data_decimated=decimate_points(data_offset,args.decimation)
      axs[idx,0].plot([d[0] for d in data_decimated],[d[1] for d in data_decimated],label=path)

  for idx,k in enumerate(args.y):
    axs[idx,0].set_ylabel(k)
  axs[0,0].legend()

  print("showing")
  #fig.show()
  plt.show()
