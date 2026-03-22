import torch
import numpy as np
import copy

from .model_utils import f_get_linear_layers, f_get_Hadamard_layers, f_get_batch
from .utils import f_col_ones, f_second_smallest, f_number_of_edges
from .layers import MaskedLinearLayer, MaskedLinearLayer2

################################################################################
################################################################################ 

def f_prune_by_L1(model):
  """Find layer and weight index of the weight with the smallest L1 norm"""

  # Get model parameters and store them in W:
  W = []
  for param in model.parameters():
    W.append(param.data.cpu().detach().numpy())

  min_weights_per_layer = [] # Array of minimum weight in each layer
  for w_layer_i_t in W: # For each layer
    w_layer_i = abs(np.array(w_layer_i_t)) # Get absolute value of weights for each layer and convert to numpy
    num_nonzero = np.sum(w_layer_i>0) # Number of nonzero weights in given layer
    # If there is a nonzero weight in this layer:
    # if num_nonzero > 1:
    if num_nonzero > 0:
      # Append the minimum absolute-value weight in the layer to min_weights_per_layer
      min_weights_per_layer.append(np.min(w_layer_i[np.where(w_layer_i>0)]))
    else:
      # Else, append np.Infinity
      min_weights_per_layer.append(np.Infinity)

  # if np.min(min_weights_per_layer) == np.Infinity:
  if f_second_smallest(min_weights_per_layer) == np.Infinity:
    # If every layer has 0 connections, the second smallest element of nonzero_weights is np.Infinity
    layer_idx = np.Infinity # Return np.Infinity, as a stopping condition (don't prune anything)
  else:
    # Else, return the index of the layer containing the minimum absolute-value weight
    layer_idx = np.argmin(np.array(min_weights_per_layer))

  prune_layer = f_get_linear_layers(model)[layer_idx] # Get layer to prune
  # prune_layer = f_get_Hadamard_layers(model)[layer_idx] # Get layer to prune
  mask = prune_layer.mask.cpu().detach().numpy() # Convert mask to numpy
  weight = prune_layer.weight.cpu().detach().numpy() # Convert weight array to numpy
  mask_idx = np.where(mask == True) # Indices of nonzeros in mask
  weight_idx = np.argmin(abs(weight[mask_idx]))
  weight_idx = tuple(np.vstack(mask_idx)[:,weight_idx]) # Index of minimum absolute-value weight

  return layer_idx, weight_idx # Return layer containing minimum weight

################################################################################
################################################################################ 

def f_trigger_pruning(log_mean_loss,min_log_mean_loss,epochs_no_improve,n_epochs_stop,epoch_dec=0.999):
  """Pause training and trigger pruning"""

  # Log of the mean loss for each training epoch
  # If the log-mean loss has decreased by at least a factor of epoch_dec:
  if log_mean_loss <= epoch_dec*min_log_mean_loss:
    epochs_no_improve = 0 # Reset counter of epochs without improvement
    min_log_mean_loss = log_mean_loss # Update the min log-mean loss
  else:
    epochs_no_improve += 1 # Increment the counter of epochs without improvement

  # If n_epochs_stop have elapsed without improvement:
  if epochs_no_improve == n_epochs_stop:
    trigger_prune = True # Set early-stop indicator to True
  else:
    trigger_prune = False # Set early-stop indicator to False

  return epochs_no_improve, min_log_mean_loss, trigger_prune

################################################################################
################################################################################ 

# def f_display_weights(model):
#   """Print model weights"""

#   eps = 1e-05

#   linear_layers = f_get_linear_layers(model)
#   print(" ")
#   print("WEIGHTS:")
#   print("Linear Layers:")
#   i = 0
#   for layer in linear_layers:
#     print("Layer:",i)
#     # print("Weights:",layer.weight)
#     print("Weights:", layer.weight/model.std_dev)

#     i+=1

#   hadamard_layers = f_get_Hadamard_layers(model)
#   print(" ")
#   print("Hadamard Layers:")
#   i = 0
#   for layer in hadamard_layers:
#     print("Layer:",i)
#     print("Weights:",layer.weight)

#     i+=1

def f_display_weights(model):
    """
    Print model weights, normalized by the model's standard deviation 
    if available, to help interpret the learned SINDy coefficients.
    """

    scaling_factor = getattr(model, 'std_dev', 1.0)
    
    eps = 1e-05

    # Display Linear Layers
    linear_layers = f_get_linear_layers(model)
    print(" ")
    print("WEIGHTS:")
    print("Linear Layers:")
    for i, layer in enumerate(linear_layers):
        print(f"Layer: {i}")
        print("Weights:", layer.weight / scaling_factor)

    # Display Hadamard Layers
    hadamard_layers = f_get_Hadamard_layers(model)
    print(" ")
    print("Hadamard Layers:")
    for i, layer in enumerate(hadamard_layers):
        print(f"Layer: {i}")
        print("Weights:", layer.weight)

        