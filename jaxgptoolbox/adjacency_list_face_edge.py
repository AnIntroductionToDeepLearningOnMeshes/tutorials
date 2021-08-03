import numpy as np
import jax
from . adjacency_edge_face import adjacency_edge_face

def adjacency_list_face_edge(F):
  E2F, E = adjacency_edge_face(F)
  adjList = np.zeros((F.shape[0], 3), dtype=np.int) # assume manifold
  E2F_bool = E2F.astype('bool')

  idx = np.arange(E.shape[0])

  for f in range(F.shape[0]):
    E2F_bool_col = np.squeeze(np.asarray(E2F_bool[:,f].todense()))
    adjList[f,:] = idx[E2F_bool_col]

  return jax.numpy.asarray(adjList)