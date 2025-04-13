import os

def checkpoint_path(checkpoint_dir,step):
  return os.path.join(checkpoint_dir,"%09d"%step)

def checkpoint_list(checkpoint_dir):
  if os.path.exists(checkpoint_dir):
    items=[]
    for i in sorted(os.listdir(checkpoint_dir)):
      try:
        items.append(int(i))
      except:
        pass
    return items
  return []
