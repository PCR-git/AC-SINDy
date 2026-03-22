import torch
import numpy as np

from .model_utils import f_get_linear_layers
from .layers import MaskedLinearLayer, MaskedLinearLayer2, HadamardLayer

################################################################################
################################################################################

def f_get_num_iters(args):
  """Get number of iterations per epoch"""
  if args.batch_mode == 1: # If using full trajectory for batch
    if args.training_mode == 0: # If using one-step lookahead
      num_iters = int(np.floor(args.data_size/args.batch_size)) - 1 # Compute batch size
    else: # If using Neural ODE
      num_iters = int(np.floor((args.data_size - args.batch_time + 1)/args.batch_size)) - 1 # Compute batch size
  else: # If using random batch
    num_iters = args.niters # Batch size = prespecified number
  return num_iters

################################################################################
################################################################################

def f_running_avg(input,avg,momentum=0.9):
  """Compute running average"""
  return avg*momentum + input*(1-momentum)

################################################################################
################################################################################  

def f_second_smallest(numbers):
  """Get second-smallest number in an array"""
  m1 = m2 = float('inf')
  for x in numbers:
      if x <= m1:
          m1, m2 = x, m1
      elif x < m2:
          m2 = x
  return m2

# def f_get_linear_layers(model):
#   """Get list of linear layers in model"""
#   linear_layers = []
#   for layer in model.children():
#     if(type(layer)==MaskedLinearLayer):
#       linear_layers.append(layer)
#   return linear_layers

################################################################################
################################################################################ 

def f_number_of_edges(model):
  """
  Get the total number of (pruneable) edges in the model
  (i.e. those in the linear layers)
  """

  num_edges = 0
  linear_layers = f_get_linear_layers(model)
  for layer in linear_layers:
    num_edges += torch.numel(layer.weight)

  return num_edges

################################################################################
################################################################################ 

def f_col_ones(x, args):
    """Concatenate column of ones to the coordinate matrix x"""
    ones = torch.ones(args.data_size, device=x.device).reshape(args.data_size, 1, 1)
    return torch.cat((ones, x), 2).to(args.device)
