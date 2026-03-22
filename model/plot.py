import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from .model_utils import f_get_linear_layers, f_get_Hadamard_layers

################################################################################
################################################################################

def f_plot_velocities(model, y_true, args):
  """Predict true and predicted velocities"""

  with torch.no_grad():
    v_pred = model.forward([],y_true[:-1]).detach().cpu().numpy().squeeze().T[1:] # Get predictions

  v_euler = ((y_true[1:] - y_true[:-1])/args.dt).detach().cpu().numpy().squeeze().T[1:] # Euler approx of velocity

  # Plot velocities in all dimensions
  for i in np.arange(np.shape(v_euler)[0]):
    fig = plt.figure()
    plt.plot(v_euler[i])
    plt.plot(v_pred[i],'k--')
    plt.legend(["Ground Truth", "Prediction"], loc ="lower right")
    plt.grid()
    plt.show()
    
################################################################################
################################################################################

def f_plot_trajectories(model,y_true,t):
  """Plot ground truth and predicted trajectories"""

  y0 = y_true[0].reshape(1,1,len(y_true.T)) # Initial position
  with torch.no_grad():
    y_pred = odeint(model, y0, t) # Get predictions

  # Reformat for plotting:
  y_true_np = y_true.detach().cpu().numpy().squeeze()
  y_pred_np = y_pred.detach().cpu().numpy().squeeze()

  # Generate 2D plot:
  if len(y_true.T) - 1 == 2:
    fig = plt.figure()
    plt.plot(y_true_np.T[1], y_true_np.T[2])
    plt.plot(y_pred_np.T[1], y_pred_np.T[2],'k--')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(["Ground Truth", "Prediction"], loc ="lower right")
    plt.grid()
    plt.show()
  # Generate 3D plot:
  elif len(y_true.T) - 1 == 3:
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(y_true_np.T[1], y_true_np.T[2], y_true_np.T[3])
    ax.plot3D(y_pred_np.T[1], y_pred_np.T[2], y_pred_np.T[3])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(["Ground Truth", "Prediction"], loc ="lower right")
    plt.grid()
    plt.show()
  else:
    print("Data needs to be 2 or 3 dimensional.")
    
################################################################################
################################################################################

# def f_plot_weights(model):
#   parameters = []
#   for parameter in model.parameters():
#     parameters.append(parameter.view(-1))
#   params = torch.concat(parameters).detach().cpu().numpy()
#   sorted_params = np.sort(params)
#   abs_params = np.abs(params)
#   sorted_abs_params = np.sort(abs_params)

#   # plt.yscale("log")
#   # plt.scatter(np.arange(len(params)),sorted_abs_params)

#   plt.scatter(np.arange(len(params)),sorted_abs_params)
#   plt.grid()
#   plt.show()

def f_plot_weights(model):

  l_weights = []
  h_weights = []

  for layer in f_get_linear_layers(model):
    l_weights.append(layer.weight.view(-1))

  for layer in f_get_Hadamard_layers(model):
    h_weights.append(layer.weight.view(-1))

  if l_weights != []:
    l_weights = torch.concat(l_weights).detach().cpu().numpy()
    plt.scatter(np.arange(len(l_weights)),np.abs(l_weights),c='b')
  if h_weights != []:
    h_weights = torch.concat(h_weights).detach().cpu().numpy()
    plt.scatter(np.arange(len(h_weights)),np.abs(h_weights),c='r',marker='x')
  plt.grid()
  plt.show()

################################################################################
################################################################################

def f_plot(input,xlabel='Iteration',ylabel='Loss',title='Training Loss',color="black"):
  """General plotting function"""
  plt.plot(input, '-', color = color)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid()
  # plt.show()

################################################################################
################################################################################

