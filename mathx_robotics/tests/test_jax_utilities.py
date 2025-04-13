from mathx.learning import jax_utilities
import math
import scipy.stats

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
      print(f"[{i}]={x}")

  p_cumulative=0
  for i in range(num_buckets):
    p_sample=float(buckets[i])/num_samples
    p_pdf=pdf(low+dx*i)*dx
    p_cumulative+=p_pdf
    
    assert abs(p_sample-p_pdf)<tolerance

  p0=1-p_cumulative
  assert abs(bucket0/num_samples-p0)<tolerance
    
def discrete_rng_test_helper(rng,p,num,num_samples=10**5,tolerance=1e-2):
  buckets=[0]*num
  
  for i in range(num_samples):
    x=rng()
    buckets[x]+=1
    
    if i%(num_samples/10)==0:
      print(f"[{i}]={x}")
    
  for i in range(num):
    p_sample=float(buckets[i])/num_samples
    p_p=p(i)
    
    assert abs(p_sample-p_p)<tolerance

def test_jax_utilities_generator():
  generator=jax_utilities.Generator(54324)
  
  continuous_rng_test_helper(rng=lambda:generator.uniform(low=-1.5,high=.75),
                             pdf=lambda x:1./(.75-(-1.5)),
                             low=-1.5,high=.75)

  continuous_rng_test_helper(rng=lambda:generator.normal(loc=1.2,scale=2.3),
                             pdf=lambda x:scipy.stats.norm.pdf(x,loc=1.2,scale=2.3),
                             low=1.2-3*2.3,high=1.2+3*2.3)

  discrete_rng_test_helper(rng=lambda:generator.integers(low=0,high=6),
                           p=lambda x:float(1)/6,
                           num=6)
