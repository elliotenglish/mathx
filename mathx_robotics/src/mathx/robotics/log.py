import logging
import sys
import os
import time

def initialize(log_file=None,append=False):
  print(f"initializing logging log_file={log_file} append={append}")
  handlers=[logging.StreamHandler(sys.stdout)]
  if log_file is not None:
    if not append and os.path.exists(log_file):
      os.remove(log_file)
    handlers.append(logging.FileHandler(log_file,mode="a"))
  #asctime = time, default = Y-M-D H:M:S,MS
  #name = root?
  #levelname = e.g. info
  #message = the formatted message
  # timestamp_format="%S.%f"
  timestamp_format=None
  formatter=logging.Formatter('[%(asctime)s] %(message)s',datefmt=timestamp_format)
  formatter.converter=time.gmtime
  for h in handlers:
    h.setFormatter(formatter)
  logging.basicConfig(
    handlers=handlers,
    level=logging.INFO,
    force=True)

def info(*args):
  logging.info(*args)

def record(**kwargs):
  info(" ".join([f"{k}={v}" for k,v in kwargs.items()]))
