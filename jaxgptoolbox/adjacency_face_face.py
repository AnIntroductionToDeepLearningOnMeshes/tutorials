import numpy as np
import scipy

def adjacency_face_face(F):
  F12 = F[:, np.array([1,2])]
  F20 = F[:, np.array([2,0])]
  F01 = F[:, np.array([0,1])]
  EAll = np.concatenate( (F12, F20, F01), axis = 0)

  EAll_sortrow = np.sort(EAll, axis = 1)
  # EAll_sortrow = uE[IC,:]
  uE, IC = np.unique(EAll_sortrow, return_inverse=True, axis=0)

  row = IC
  col = np.tile(np.arange(F.shape[0]), 3)
  val = np.ones(len(IC), dtype=np.int)
  uE2F = scipy.sparse.coo_matrix((val,(row, col)), shape=(uE.shape[0], F.shape[0])).tocsr()
  A = (uE2F.T * uE2F) > 0

  return A