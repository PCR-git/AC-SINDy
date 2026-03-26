import torch
import numpy as np

from .layers import MaskedLinearLayer, MaskedLinearLayer2, HadamardLayer

################################################################################
################################################################################  

# Divide layer
def f_smooth_divide(x,eps=1e-5):
  return torch.tanh(x/eps)/(torch.abs(x)+eps)

################################################################################
################################################################################  

# def f_get_linear_layers(model):
  # """Get list of linear layers in model"""
  # linear_layers = []
  # for layer in model.children():
  #   if(type(layer)==MaskedLinearLayer):
  #     linear_layers.append(layer)
  # return linear_layers

def f_get_linear_layers(model):
  """Get list of linear layers in model"""
  linear_layers = []
  for layer in model.modules():
    if type(layer)==MaskedLinearLayer or type(layer)==MaskedLinearLayer2:
      linear_layers.append(layer)
  return linear_layers

################################################################################
################################################################################  

def f_get_Hadamard_layers(model):
  """Get list of Hadamard layers in model"""
  hadamard_layers = []
  for layer in model.modules():
    if(type(layer)==HadamardLayer):
      hadamard_layers.append(layer)
  return hadamard_layers

################################################################################
################################################################################ 

def f_get_batch(model,y_true,itr,args):
  """Get a batch of training data"""

  # Indices of batch:
  if args.batch_mode == 0: # Draw batch randomly
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
  else: # Sequentially sample portion of entire simulated trajectory
    s = torch.tensor(itr*args.batch_size + np.arange(args.batch_size))

  batch_y0 = y_true[s].to(args.device) # Batch of inputs
  if args.training_mode == 0: # If using one-step look ahead
    batch_t = [] # Don't need batch_t
    batch_y = y_true[s+1].to(args.device) # Ground-truth of output
  else: # If using Neural ODE
    batch_t = t[:args.batch_time].to(args.device) # Array of time steps, starting from zero
    batch_y = torch.stack([y_true[s + i] for i in range(args.batch_time)], dim=0).to(args.device) # Ground-truth of outputs

  # batch_y0, model.std_dev = f_batchnorm(batch_y0,eps=1e-05,layer=0)
  # batch_y = f_batchnorm2(batch_y,model.std_dev,eps=1e-05)

  # f_update_weights(self,eps=1e-05)
  # self.prev_std_dev = self.std_dev
  # inputs, self.std_dev = f_batchnorm(inputs,eps=1e-05,layer=0)

  return batch_y0, batch_t, batch_y

################################################################################
################################################################################  

# L1 loss:
def f_L1_loss(model,l1_weight=1):
    # Get model parameters:
    parameters = []
    for parameter in model.parameters():
      parameters.append(parameter.view(-1))
    return l1_weight*torch.abs(torch.cat(parameters)).sum()

# L1 loss on linear weights:
def f_linear_L1_loss(model,l1_weight=0):
    # Get model parameters:
    weights = []
    linear_layers = f_get_linear_layers(model)
    for layer in linear_layers:
      weights.append(layer.weight.view(-1))
    return l1_weight*torch.abs(torch.cat(weights)).sum()

# L1 loss on Hadamard weights:
def f_Hadamard_L1_loss(model,l1_weight=0):
    # Get model parameters:
    weights = []
    hadamard_layers = f_get_Hadamard_layers(model)
    for layer in hadamard_layers:
      weights.append(layer.weight.view(-1))
    return l1_weight*torch.abs(torch.cat(weights)).sum()

# # L2 loss:
# def f_L2_loss(w,l2_weight=1):
#   # Get model parameters:
#   parameters = []
#   for parameter in model.parameters():
#     parameters.append(parameter.view(-1))
#   return l2_weight*torch.square(torch.cat(parameters)).sum()

# # Thresholded L1 loss:
# def f_Thresholded_L1_loss(model,l1_weight=1):
#   # Get model parameters:
#   parameters = []
#   for parameter in model.parameters():
#     parameters.append(parameter.view(-1))
#   abs_vals = torch.abs(torch.cat(parameters))
#   idx = torch.where(abs_vals > 0)
#   threshold = 2*torch.min(abs_vals[idx])*len(idx[0])
#   idx2 = torch.where(abs_vals > threshold)
#   abs_vals[idx2] = threshold
#   return l1_weight*abs_vals.sum()

################################################################################
################################################################################  

# def f_batchnorm(sample,eps=1e-05):
#     # mu = torch.mean(sample,axis=0)
#     mu = 0
#     var = torch.var(sample,axis=0)
#     return (sample-mu)/torch.sqrt(var+eps), var

# s = torch.tensor(1*args.batch_size + np.arange(args.batch_size))
# sample = y_true[s]
# sample.size()

# sample
# f_batchnorm(sample)
# sample[:,:,1:3]

################################################################################
################################################################################  

def f_layer_norm(input,eps=1e-5):
    mu = torch.mean(input,axis=2,keepdims=True)
    # var = torch.var(input,axis=2,keepdims=True)
    # return (input - mu)/torch.sqrt(var+eps)
    return input - mu

################################################################################
################################################################################ 

def f_batchnorm(batch,eps=1e-05,layer=1):
    if layer == 0:
      batch = batch[:,:,1:]
    # mu = torch.mean(batch,axis=0)
    mu = 0

    var = torch.var(batch,axis=0)[0]

    std_dev = torch.sqrt(var+eps)

    batch_norm = (batch-mu)/std_dev

    if layer == 0:
      batch_norm = torch.cat((batch[:,:,0:1],batch_norm),2)
      std_dev = torch.cat((torch.tensor([1]),std_dev))

    return batch_norm, std_dev

def f_batchnorm2(batch,std_dev,eps=1e-05):

    mu = 0
    batch_norm = (batch-mu)/std_dev

    return batch_norm

################################################################################
################################################################################ 

def f_update_weights(model,eps):
  """Get model weights"""

  linear_layers = f_get_linear_layers(model)
  # print(" ")
  # print("WEIGHTS:")
  # print("Linear Layers:")
  i = 0
  for layer in linear_layers:
    # print("Layer:",i)
    # print("Weights:",layer.weight)
    with torch.no_grad():
      layer.weight *= model.std_dev/model.prev_std_dev
    # print(layer.weight)

    i+=1

  hadamard_layers = f_get_Hadamard_layers(model)
  # print("Hadamard Layers:")
  i = 0
  for layer in hadamard_layers:
    # print("Layer:",i)
    # print("Weights:",layer.weight)
    with torch.no_grad():
      layer.weight *= model.std_dev/model.prev_std_dev
    # print(layer.weight)

    i+=1

################################################################################
################################################################################ 

def f_prepare_rnn_data(y_data, history_len, future_len):
    """
    Converts trajectory into (Batch, History+Future, Dim)
    """
    n_points = y_data.shape[0]
    windows = []
    for i in range(n_points - history_len - future_len):
        windows.append(y_data[i : i + history_len + future_len])
    
    # Shape: [Num_Windows, History + Future, Samples, Dim]
    return torch.stack(windows)


