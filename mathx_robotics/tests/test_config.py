import os

def debug_visualize():
  return os.environ.get("DEBUG_VISUALIZE",None)=="1"

def debug_write():
  return os.environ.get("DEBUG_WRITE",None)=="1"

def debug_value(key,default_value):
  value=os.environ.get(key,str(default_value))
  return type(default_value)(value)
