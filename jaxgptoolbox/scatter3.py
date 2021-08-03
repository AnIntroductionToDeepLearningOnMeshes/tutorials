import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter3(P):
  fig = plt.figure()
  # ax = Axes3D(fig)
  ax = plt.axes(projection='3d')
  ax.scatter(P[:,0], P[:,1], P[:,2])
  ax.axis('equal')