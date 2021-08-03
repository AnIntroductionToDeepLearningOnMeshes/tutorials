import numpy as np
import scipy
import jax
from . edges_with_mapping import edges_with_mapping

def adjacency_edge_face(F):
  uE, F2E = edges_with_mapping(F)
  IC = F2E.T.reshape(F.shape[1]*F.shape[0])

  row = IC
  col = np.tile(np.arange(F.shape[0]), 3)
  val = np.ones(len(IC), dtype=np.int)
  uE2F = scipy.sparse.coo_matrix((val,(row, col)), shape=(uE.shape[0], F.shape[0])).tocsr()

  return uE2F, jax.numpy.asarray(uE)
