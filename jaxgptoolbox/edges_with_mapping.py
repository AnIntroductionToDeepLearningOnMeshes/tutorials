import numpy as np
import jax

def edges_with_mapping(F):
  F12 = F[:, np.array([1,2])]
  F20 = F[:, np.array([2,0])]
  F01 = F[:, np.array([0,1])]
  EAll = np.concatenate( (F12, F20, F01), axis = 0)

  EAll_sortrow = np.sort(EAll, axis = 1)
  # EAll_sortrow = uE[F2E,:]
  uE, F2E_vec = np.unique(EAll_sortrow, return_inverse=True, axis=0)
  F2E = F2E_vec.reshape(F.shape[1], F.shape[0]).T
  return uE, F2E