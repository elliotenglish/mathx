def generate_test_equilibrium():
  surf = FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[
      (0, 0),
      (1, 0),
      (1, 1),
      (-1, -1),
    ],  # (m,n) pairs corresponding to R_mn on previous line
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=5,
  )

  pressure = PowerSeriesProfile(
    [1.8e4, 0, -3.6e4, 0, 1.8e4]
  )  # coefficients in ascending powers of rho
  iota = PowerSeriesProfile([1, 0, 1.5])  # 1 + 1.5 r^2

  eq_init = Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface=surf,
    pressure=pressure,
    iota=iota,
    Psi=1.0,  # total flux, in Webers
  )

  eq_sol, info = eq_init.solve(verbose=3, copy=True)
  
  return eq_sol

class Plasma:
  def __init__(self,params):
    pass
  
  def get_surface(self,x):
    pass
  

def get_test_plasma(path="equilibrium.h5"):

