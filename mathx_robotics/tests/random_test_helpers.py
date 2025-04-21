import math

import mathx.core.log as log

def continuous_rng_test_helper(rng,pdf,low,high,num_buckets=100,num_samples=10**5,tolerance=1e-2):
  dx=(high-low)/num_buckets

  bucket0=0
  buckets=[0]*num_buckets

  for i in range(num_samples):
    x=rng()
    idx=math.floor((x-low)/dx)
    if idx>=0 and idx<num_buckets:
      buckets[idx]+=1
    else:
      bucket0+=1
    if i%(num_samples/10)==0:
      log.info(f"[{i}]={x}")
      
  log.info(f"sum={sum(buckets)+bucket0} num_samples={num_samples}")

  p_cumulative=0
  for i in range(num_buckets):
    p_sample=float(buckets[i])/num_samples
    x=low+dx*i
    p_pdf=pdf(x)*dx
    p_cumulative+=p_pdf
    log.info(f"p[{x},{x+dx}] sample={p_sample} pdf={p_pdf}")

    # assert abs(p_sample-p_pdf)<tolerance

  p_sample0=bucket0/num_samples
  p_pdf0=1-p_cumulative
  log.info(f"p[-] sample={p_sample0} pdf={p_pdf0}")
  assert abs(p_sample0-p_pdf0)<tolerance
    
def discrete_rng_test_helper(rng,p,num,num_samples=10**5,tolerance=1e-2):
  buckets=[0]*num
  
  for i in range(num_samples):
    x=rng()
    buckets[x]+=1
    
    if i%(num_samples/10)==0:
      log.info(f"[{i}]={x}")
    
  for i in range(num):
    p_sample=float(buckets[i])/num_samples
    p_p=p(i)
    
    assert abs(p_sample-p_p)<tolerance
