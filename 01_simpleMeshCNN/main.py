import sys
sys.path.append('..')

import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp 
import numpy.random as random
import glob, time, pickle

import jaxgptoolbox as jgp

random.seed(0) # set random seed

def compute_input(V,F,E,flap):
  # dihedral angles
  FN = jgp.faceNormals(V,F)
  E2F = jgp.adjacency_list_edge_face(F)
  dotN = np.sum(FN[E2F[:,0],:] * FN[E2F[:,1],:], axis = 1).clip(-1, 1) # clip to avoid nan
  dihedral_angles = (np.pi - np.arccos(dotN)) / np.pi

  # edge length ratios
  Elen = np.sqrt(np.sum((V[E[:,0],:] - V[E[:,1],:])**2, axis = 1))
  nE = E.shape[0]
  ratio_e_a = Elen[flap[:,0]] / Elen[np.arange(nE)] 
  ratio_e_b = Elen[flap[:,1]] / Elen[np.arange(nE)] 
  ratio_e_c = Elen[flap[:,2]] / Elen[np.arange(nE)] 
  ratio_e_d = Elen[flap[:,3]] / Elen[np.arange(nE)] 
  ratio_features = np.stack((ratio_e_a, ratio_e_b, ratio_e_c, ratio_e_d), axis = 1)
  ratio_features = np.sort(ratio_features, axis = 1)

  # concatenate
  dihedral_angles = np.expand_dims(dihedral_angles,1) 
  fE = np.concatenate((dihedral_angles,ratio_features), axis = 1) 
  return fE # fE.shape = (#edges, #channels)

def get_flap_edges(F):
  E, F2E = jgp.edges_with_mapping(F)
  flapEdges = [[] for i in range(E.shape[0])]
  for f in range(F.shape[0]):
    e0 = F2E[f,0]
    e1 = F2E[f,1]
    e2 = F2E[f,2]
    flapEdges[e0].extend([e1,e2])
    flapEdges[e1].extend([e2,e0])
    flapEdges[e2].extend([e0,e1])
  return np.array(flapEdges), E

trainFolder = '../datasets/meshMNIST_100V/12_meshMNIST/' # replace this with the path to the dataset folder
labels = np.asarray(onp.loadtxt(trainFolder + 'labels.txt', delimiter=','), dtype=np.int16) # replace this with the path to the label file

numMeshes = len(glob.glob(trainFolder + "*.obj"))
meshList = [{} for sub in range(numMeshes)]
print("this will take a while, so we should put it into preprocessing")
for meshIdx in range(numMeshes):
  if meshIdx % 100 == 0:
    print(str(meshIdx) + "/" + str(numMeshes))
  meshName = str(meshIdx+1).zfill(4) + ".obj"
  V,F = jgp.readOBJ(trainFolder + meshName)
  flap, E = get_flap_edges(F)
  fE = compute_input(V,F,E,flap)

  meshList[meshIdx]["V"] = V
  meshList[meshIdx]["F"] = F
  meshList[meshIdx]["flap"] = flap
  meshList[meshIdx]["fE"] = fE
  meshList[meshIdx]["label"] = labels[meshIdx,1] - 1.0 # 0: "1", 1: "2"

def initialize_meshCNN_weights(conv_dims, mlp_dims):
  num_flap_edges = 5
  scale = 1e-2

  # Conv filter parameters
  params_conv = []
  for ii in range(len(conv_dims) - 1):
    C = scale * random.randn(conv_dims[ii+1], conv_dims[ii], num_flap_edges)
    params_conv.append(C)

  # MLP parameters
  params_mlp = []
  W = scale * random.randn(mlp_dims[0], conv_dims[-1])
  b = scale * random.randn(mlp_dims[0])
  params_mlp.append([W, b])
  for ii in range(len(mlp_dims) - 1):
    W = scale * random.randn(mlp_dims[ii+1], mlp_dims[ii])
    b = scale * random.randn(mlp_dims[ii+1])
    params_mlp.append([W, b])

  params = (params_conv, params_mlp)
  return params

conv_dims = [5,8,8]
mlp_dims = [4,1]
params = initialize_meshCNN_weights(conv_dims, mlp_dims)

def forward(fE, params, flap):
  # =========================
  # edge functions (fE) to flap functions (fP)
  def E2P(fE, flap):
    # fE.shape = (#edges, #channels)
    # fP.shape = (#edges, #channels, 5)
    e = fE[np.arange(flap.shape[0]),:]
    a = fE[flap[:,0],:]
    b = fE[flap[:,1],:]
    c = fE[flap[:,2],:]
    d = fE[flap[:,3],:]
    fP = np.stack((e, np.abs(a-c), a+c, np.abs(b-d), b+d), axis = 2)
    return fP

  # mesh convolution
  def mesh_conv_single(W, fP_i):
    # fP_i.shape = (#input_channels, 5)
    # W.shape = (#output_channels, #input_channels 5)
    dot = W * np.expand_dims(fP_i, 0)
    fE = np.sum(np.sum(dot,1),1)
    return fE
  # vectorize "mesh_conv_single" so that it can process all fP
  mesh_conv = jax.vmap(mesh_conv_single, in_axes=(None, 0), out_axes=0)
  # =========================

  params_conv = params[0]
  params_mlp = params[1]
  
  # mesh convolutions
  for ii in range(len(params_conv)):
    conv_W = params_conv[ii]
    fP = E2P(fE, flap)
    fE = mesh_conv(conv_W, fP)
    fE = jax.nn.relu(fE)

  # global pooling (max pooling)
  f = np.max(fE,0)

  # fully connected
  for ii in range(len(params_mlp)):
    W, b = params_mlp[ii]
    f = np.dot(W, f) + b
    if ii == (len(params_mlp) - 1):
      f = jax.nn.sigmoid(f)
    else:
      f = jax.nn.relu(f)
  return f

pred = forward(meshList[0]['fE'], params, meshList[0]['flap'])
print(pred)

def loss(params, flap, input, label):
  pred = forward(input, params, flap)[0]
  return -label * np.log(pred) - (1.0-label) * np.log(1 - pred)

stepSize = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size=stepSize)
opt_state = opt_init(params)

@jit
def update(epoch, opt_state, input, label, flap):
  params = get_params(opt_state)
  value, grads = value_and_grad(loss, argnums=0)(params, flap, input, label)
  opt_state = opt_update(epoch, grads, opt_state)
  return value, opt_state

numEpochs = 200
bestLoss = np.inf
for epoch in range(numEpochs):
  ts = time.time()
  loss_total = 0.0
  for meshIdx in range(numMeshes): 
    lossVal, opt_state = update(epoch, opt_state, \
      meshList[meshIdx]['fE'], \
      meshList[meshIdx]['label'], \
      meshList[meshIdx]['flap'])
    loss_total += lossVal
  loss_total /= numMeshes

  if epoch % 1 == 0:
    print("epoch %d, train loss %f, epoch time: %.7s sec" % (epoch, loss_total, time.time() - ts))

NETPARAM = "params.pickle"
params_final = get_params(opt_state)
with open(NETPARAM, 'wb') as handle:
  pickle.dump(params_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
