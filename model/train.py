import torch
import numpy as np
import copy
from tqdm import tqdm
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim

from .model_utils import f_get_batch, f_get_linear_layers
from .utils import f_get_num_iters, f_running_avg, f_number_of_edges, f_col_ones
from .pruning import f_trigger_pruning, f_display_weights
from .plot import f_plot_velocities, f_plot_weights
# from .layers import MaskedLinearLayer, MaskedLinearLayer2, HadamardLayer

################################################################################
################################################################################

def f_get_prediction(model,batch_y0,batch_t,args):
  """Predict trajectory"""

  if args.training_mode == 0: # If using one-step lookahead
    pred_v = model.forward([], batch_y0).to(args.device) # Get velocity prediction
    pred_y = batch_y0 + pred_v*args.dt # Predict next position using Euler approx
  else: # If using Neural ODE
    pred_y = odeint(model, batch_y0, batch_t).to(args.device) # Predict next position using Neural ODE

  return pred_y

################################################################################
################################################################################

def f_train(model,losses,loss,optimizer,y_true,args):
  """Train the model for one epoch"""

  model.train() # Training mode
  epoch_losses = [] # Array to hold losses for this epoch

  num_itr = f_get_num_iters(args) # Get number of iterations
  for itr in np.arange(num_itr):
    optimizer.zero_grad() # Zero out gradients

    # Get training data batch:
    batch_y0, batch_t, batch_y = f_get_batch(model,y_true,itr,args)
    pred_y = f_get_prediction(model,batch_y0,batch_t,args) # Predict trajectory
    cur_loss = loss(batch_y,pred_y) # Compute loss at current iteration
    # cur_loss += f_linear_L1_loss(model,l1_weight=0.00005)
    # cur_loss += f_Hadamard_L1_loss(model,l1_weight=0.00005)

    cur_loss.backward() # Backprop
    optimizer.step() # Step of optimizer

    cur_loss_np = cur_loss.cpu().detach().numpy()
    losses.append(cur_loss_np) # Array of losses
    epoch_losses.append(cur_loss_np) # Array of epoch losses

  return losses, epoch_losses

################################################################################
################################################################################

def f_train_model(y_true, y_valid, t, model, loss, opt, w, args, max_epochs=500, lr=2e-3, n_epochs_stop=4, epoch_dec=1, num_to_prune=1):
    """
    Main training and pruning loop for Neural SINDy.

    This function trains the model until it nears convergence, then iteratively 
    prunes the least significant weights based on their effect on the loss function.
    The process continues until the network is sparsified into a symbolic representation.

    Args:
        y_true (Tensor): Preprocessed training trajectory (should include bias/time columns).
        y_valid (Tensor): Preprocessed validation trajectory.
        t (Tensor): Time vector associated with the trajectories.
        model (nn.Module): The Arithmetic Circuit or Sine-Product network.
        loss (function): Loss function (e.g., nn.L1Loss).
        opt (Optimizer): Torch optimizer (e.g., Adam).
        w (int): Width of the hidden layers.
        args (Namespace): Configuration arguments (device, dt, training_mode, etc.).
        max_epochs (int): Maximum training epochs.
        lr (float): Initial learning rate.
        n_epochs_stop (int): Patience for early stopping/pruning triggers.
        epoch_dec (float): Decay threshold for pruning logic.
        num_to_prune (int): Number of edges to remove per pruning event.

    Returns:
        tuple: (losses, mean_losses, smoothed_log_mean_losses, prune_epochs)
    """

    losses = []                   # Training losses per iteration
    mean_losses = []              # Mean training losses per epoch
    smoothed_log_mean_losses = [] # Running-average log losses for convergence tracking

    num_edges = f_number_of_edges(model) # Total count of learnable parameters
    layer_idx = 0
    trigger_prune = False         # Flag to initiate a pruning event
    epochs_no_improve = 0         # Counter for non-improving epochs
    min_smoothed_log_mean_loss = np.Infinity
    best_smoothed_log_mean_loss = np.Infinity
    num_pruned = 0
    prune_epochs = []             # List to track when pruning occurs

    # Main training loop
    for epoch in tqdm(np.arange(max_epochs) + 1, desc="Training progress..."):

        # Perform one epoch of training
        losses, epoch_losses = f_train(model, losses, loss, opt, y_true, args)
        mean_loss = np.mean(epoch_losses)
        mean_losses.append(mean_loss)

        # Compute log-mean loss for convergence analysis
        log_mean_loss = np.log(mean_loss)

        # Update running average of the log-mean loss
        if smoothed_log_mean_losses == []:
            smoothed_log_mean_loss = log_mean_loss
        else:
            smoothed_log_mean_loss = smoothed_log_mean_losses[-1]
        
        smoothed_log_mean_loss = f_running_avg(log_mean_loss, smoothed_log_mean_loss)
        smoothed_log_mean_losses.append(smoothed_log_mean_loss)

        # Determine if training has plateaued enough to trigger pruning
        epochs_no_improve, min_smoothed_log_mean_loss, trigger_prune = f_trigger_pruning(
            smoothed_log_mean_loss, min_smoothed_log_mean_loss, epochs_no_improve, n_epochs_stop, epoch_dec
        )

        # Save the best model weights if current performance is optimal
        if smoothed_log_mean_loss <= best_smoothed_log_mean_loss:
            best_smoothed_log_mean_loss = smoothed_log_mean_loss
            model.best_weights = copy.deepcopy(model.state_dict())
            model.best_epoch = epoch

        # Handle pruning logic when triggered
        if trigger_prune:
            # Reset convergence tracking for the new, pruned architecture
            min_smoothed_log_mean_loss = np.Infinity
            epochs_no_improve = 0 

            # Visual feedback during pruning stages
            if args.training_mode != 1: 
                f_plot_velocities(model, y_valid, args)
                f_plot_weights(model)

                prune_epochs.append(epoch)
                print(f"\n--- Pruning Triggered at Epoch {epoch} ---")
                f_display_weights(model)

                # Iteratively prune the specified number of edges
                for _ in np.arange(num_to_prune):
                    num_pruned += 1

                    # Prune the edge with the minimal impact on model output
                    layer_idx, weight_idx = f_prune_by_effect(model, y_valid, loss, args)

                    # If the network cannot be pruned further, stop training
                    if layer_idx == np.Infinity:
                        break
                    
                    # Apply the mask to the specific weight in the linear layer
                    linear_layers = f_get_linear_layers(model)
                    linear_layers[layer_idx].compute_mask(weight_idx)

                print(f"Pruned edges: {num_pruned}")
                print(f"Remaining edges: {num_edges - num_pruned}")

                # Switch training mode back to one-step-lookahead for fine-tuning
                args.training_mode = 0 

            print(f"Current Training Mode: {args.training_mode}")
            print('----------------------------------------------')

            # Reset the optimizer to clear momentum/state for the new architecture
            opt = optim.Adam(model.parameters(), lr=lr)

        # Terminate global loop if no more weights can be pruned
        if layer_idx == np.Infinity:
            break

    return losses, mean_losses, smoothed_log_mean_losses, prune_epochs

################################################################################
################################################################################ 

def f_test_pruned_model(pruned_model,loss,y_true,args):
  """Test pruned model against ground truth"""

  tot_loss = 0 # Total loss
  num_itr = f_get_num_iters(args) # Get number of iterations
  for itr in np.arange(num_itr): # For number of iterations in an epoch
    batch_y0, batch_t, batch_y = f_get_batch(pruned_model,y_true,itr,args) # Get training data batch
    pred_y = f_get_prediction(pruned_model,batch_y0,batch_t,args) # Get predicted trajectory
    cur_loss = loss(pred_y, batch_y) # Compute loss at this iteration
    tot_loss += cur_loss # Add to total loss so far

  return tot_loss/num_itr # Return average loss per iteration

################################################################################
################################################################################ 

def f_prune_by_effect(model,y_true,loss,args):
  """
  Prune the edge that, when pruned, results in the highest loss
  (The original version of this function pruned the edge with the least
  effect on the output of the network)
  """

  loss_every_layer = [] # Array to hold losses due to all edge prunings
  modelB = copy.deepcopy(model) # Make a copy of the neural network
  layers = f_get_linear_layers(modelB) # Get the linear layers of model B
  # layers = f_get_Hadamard_layers(modelB) # Get the linear layers of model B

  for layer in layers: # For all linear layers
    loss_per_layer = [] # Array to hold all losses for this layer
    mask = layer.mask.cpu().detach().numpy() # Get mask
    indices = np.where(mask>0) # Indices where mask is nonzero
    if np.shape(indices)[1] > 0: # If there is more than one unpruned edge in this layer
      for i in np.arange(len(indices[0])): # For every (unpruned) edge in the layer
        layer.mask[indices[0][i],indices[1][i]] = 0 # Drop the ith edge
        # loss_i = f_test_loss(modelB,loss).detach().numpy()
        # loss_i = f_compare_models(model,modelB,loss).detach().numpy()

        # Test the pruned model against ground truth:
        loss_i = f_test_pruned_model(modelB,loss,y_true,args).cpu().detach().numpy()
        loss_per_layer.append(loss_i) # Append loss to array
        # modelB = copy.deepcopy(model)
        layer.mask[indices[0][i],indices[1][i]] = 1 # Reset the ith edge to value 1
    else: # If there are no unpruned edges
      loss_per_layer.append(np.Infinity) # Set the loss for this layer to np.Infinity

    loss_per_layer = np.array(loss_per_layer) # Convert to numpy array
    # idx = np.argmin(loss_per_layer)
    # Append array of losses for this layer to loss_every_layer:
    loss_every_layer.append(loss_per_layer)

  # Convert loss_every_layer to dtype object
  # so that we can iterate through the array:
  loss_every_layer = np.array(loss_every_layer,dtype='object')
  # Append min losses for each layer to array:
  min_layer_losses = []
  for layer_losses in loss_every_layer:
    min_layer_losses.append(np.min(layer_losses))

  # If the min loss is np.Infinity, set layer_idx and weight_idx to np.Infinity
  # This will cause pruning to stop
  if np.min(np.concatenate(loss_every_layer)) == np.Infinity:
    layer_idx = np.Infinity
    weight_idx = np.Infinity
  else:
    # Get the index of the layer with min loss:
    layer_idx = np.argmin(min_layer_losses)
    # Get the index of the edge with min loss in this layer:
    nonzero_idx = np.argmin(loss_every_layer[layer_idx])

    print('layer_idx =',layer_idx) # Print index of layer with min loss
    print('index of pruned edge =',nonzero_idx) # Print index of pruned (min-loss) edge in this layer
    # print('min loss caused by pruning an edge =',min(loss_every_layer[layer_idx])) # Print min loss caused by pruning an edge

    layer = layers[layer_idx] # Layer with min loss
    mask = layer.mask.cpu().detach().numpy() # Mask for this layer
    indices = np.where(mask>0) # Nonzero indices
    # print("mask =",mask)
    # print("Nonzero indices =",indices)

    # Index of the minimum-loss weight:
    weight_idx = tuple((indices[0][nonzero_idx],indices[1][nonzero_idx]))

  return layer_idx, weight_idx

