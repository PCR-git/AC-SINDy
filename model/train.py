import torch
import numpy as np
import copy
from tqdm import tqdm
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .model_utils import f_get_batch, f_get_linear_layers
from .utils import f_get_num_iters, f_running_avg, f_number_of_edges, f_col_ones
from .pruning import f_trigger_pruning, f_display_weights
from .plot import f_plot_velocities, f_plot_lookahead, f_plot_weights, f_plot
from .models import AC_Filter_Net, AC_Filter_Norm_Net, AC_Filter_PreNorm_Net

################################################################################
################################################################################

# def f_get_prediction(model,batch_y0,batch_t,args):
#   """Predict trajectory"""

#   if args.training_mode == 0: # If using one-step lookahead
#     pred_v = model.forward([], batch_y0).to(args.device) # Get velocity prediction
#     pred_y = batch_y0 + pred_v*args.dt # Predict next position using Euler approx
#   else: # If using Neural ODE
#     pred_y = odeint(model, batch_y0, batch_t).to(args.device) # Predict next position using Neural ODE

#   return pred_y

def f_get_prediction(model, batch_y0, batch_t, args):
    """Predict trajectory / sequence"""

    # --- CASE 1: AC_Filter_Net (RNN/Attention) ---
    if isinstance(model, (AC_Filter_Net, AC_Filter_Norm_Net, AC_Filter_PreNorm_Net)):
        # The model already returns integrated positions [Batch, L, 8]
        # t is not used in this specific model's forward pass
        return model(None, batch_y0).to(args.device)

    # --- CASE 2: Standard SINDy (AC_Net, HLM_Net, etc.) ---
    else:
        if args.training_mode == 0: # One-step lookahead
            pred_v = model.forward([], batch_y0).to(args.device) 
            # Standard SINDy returns Velocity, so we Euler integrate here
            return batch_y0 + pred_v * args.dt 
        else: # Neural ODE
            return odeint(model, batch_y0, batch_t).to(args.device)

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step() # Step of optimizer

    cur_loss_np = cur_loss.cpu().detach().numpy()
    losses.append(cur_loss_np) # Array of losses
    epoch_losses.append(cur_loss_np) # Array of epoch losses

  return losses, epoch_losses

################################################################################
################################################################################

def f_get_batch_rnn(data, itr, args):
    start_idx = itr * args.batch_size
    # batch shape: [Batch, Seq_Len, 1, Dim] or [Batch, Seq_Len, Dim]
    batch = data[start_idx : start_idx + args.batch_size].to(args.device)
    h_len = getattr(args, 'num_t_steps', 10)
    
    if batch.dim() == 4:
        batch = batch.squeeze(2) # Standardize to [Batch, Seq_Len, Dim]
        
    # history: [Batch, 10, 3] -> Used for model input
    history = batch[:, :h_len, :]
    
    # We return 'batch' as the 3rd arg so f_train_rnn has the 'future' steps
    return history, None, batch

################################################################################

# def f_train_rnn(model, losses, loss_fn, optimizer, y_true, args):
#     model.train()
#     epoch_losses = []
#     num_itr = f_get_num_iters(args)
#     h_len = args.num_t_steps # 10
#     f_len = args.batch_time   # 4

#     for itr in range(num_itr):
#         optimizer.zero_grad()
#         batch_y0, _, batch_full = f_get_batch_rnn(y_true, itr, args)

#         # 1. Predict: [Batch, 10, 8]
#         pred_y_seq = model(None, batch_y0) 

#         # 2. Align Target: 
#         # batch_full is [Batch, 14, 3]. We need the (x,y) for 10 windows of 4-steps.
#         y_feats = batch_full[:, :, 1:] # [Batch, 14, 2]
        
#         # This creates sliding windows of the future
#         target_windows = y_feats.unfold(1, f_len, 1) # [Batch, 11, 2, 4]
        
#         # Transpose to [Batch, 11, 4, 2] to match feature stacking
#         target_windows = target_windows.transpose(2, 3)
        
#         # We want windows starting at index 1 through 10
#         target_y = target_windows[:, 1:h_len+1, :, :].reshape(batch_full.shape[0], h_len, -1)

#         # 3. Loss
#         cur_loss = loss_fn(pred_y_seq, target_y)
#         cur_loss.backward()
#         optimizer.step()

#         epoch_losses.append(cur_loss.item())
        
#     return losses, epoch_losses

# def f_train_rnn(model, losses, loss_fn, optimizer, y_true, args, use_norm=False):
#     """
#     Corrected Training Function.
#     The model (AC_Filter_Norm_Net) now handles internal scaling. 
#     Therefore, we compare Real-World Predictions to Real-World Targets.
#     """
#     model.train()
#     epoch_losses = []
#     num_itr = f_get_num_iters(args)
#     h_len = args.num_t_steps # Typically 10
#     f_len = args.batch_time   # Typically 4

#     for itr in range(num_itr):
#         optimizer.zero_grad()
        
#         # 1. Get Batch: batch_y0 [Batch, 10, 3], batch_full [Batch, 14, 3]
#         batch_y0, _, batch_full = f_get_batch_rnn(y_true, itr, args)

#         # 2. Forward Pass
#         # Even though the model uses norm internally, the return is scaled 
#         # back to original units (e.g., x ~ 10.0 instead of x ~ 1.0)
#         pred_y_seq = model(None, batch_y0) 

#         # 3. Target Alignment
#         # Extract physical features (drop bias column 0)
#         y_feats = batch_full[:, :, 1:] # Shape [Batch, 14, 2]
        
#         # Create sliding windows of the future
#         target_windows = y_feats.unfold(1, f_len, 1).transpose(2, 3)
        
#         # Align target to the model's prediction window [Batch, 10, 8]
#         # These are RAW coordinates (Real World)
#         target_y = target_windows[:, 1:h_len+1, :, :].reshape(batch_full.shape[0], h_len, -1)
        
#         # 4. Loss & Backprop
#         # Both tensors are now in the same coordinate system
#         cur_loss = loss_fn(pred_y_seq, target_y)
#         cur_loss.backward()
        
#         # Gradient clipping to stabilize the Arithmetic Circuit multiplicative logic
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
#         epoch_losses.append(cur_loss.item())
        
#     return losses, epoch_losses

# training.py

def f_train_rnn(model, losses, loss_fn, optimizer, y_true, args, use_norm=False):
    # use_norm is kept in the signature for API compatibility but no longer
    # needs to do anything — AC_Filter_Norm_Net handles scaling internally
    model.train()
    epoch_losses = []
    num_itr = f_get_num_iters(args)
    h_len, f_len = args.num_t_steps, args.batch_time

    for itr in range(num_itr):
        optimizer.zero_grad()
        batch_y0, _, batch_full = f_get_batch_rnn(y_true, itr, args)

        # Forward pass — always physical scale, regardless of model type
        pred_y_seq = model(None, batch_y0)

        # Target — physical scale, no conditional normalization needed
        y_feats = batch_full[:, :, 1:]
        target_windows = y_feats.unfold(1, f_len, 1).transpose(2, 3)
        target_y = target_windows[:, 1:h_len+1, :, :].reshape(
            batch_full.shape[0], h_len, -1
        )

        cur_loss = loss_fn(pred_y_seq, target_y)
        cur_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_losses.append(cur_loss.item())

    return losses + [l for l in epoch_losses], epoch_losses

################################################################################
################################################################################

# def f_train_model(y_true, y_valid, t, model, loss, opt, w, args, 
#                   max_epochs=500, lr=2e-3, n_epochs_stop=4, 
#                   epoch_dec=1, num_to_prune=1, prune_mode='gradient'):
#     """
#     Main training and pruning loop for Neural SINDy.
    
#     prune_mode: 'gradient' (Scalable) or 'effect' (Brute-force accurate)
#     """

#     losses = []
#     mean_losses = []
#     smoothed_log_mean_losses = []

#     num_edges = f_number_of_edges(model)
#     layer_idx = 0
#     trigger_prune = False
#     epochs_no_improve = 0
#     min_smoothed_log_mean_loss = np.Infinity
#     best_smoothed_log_mean_loss = np.Infinity
#     num_pruned = 0
#     prune_epochs = []

#     for epoch in tqdm(np.arange(max_epochs) + 1, desc="Training progress..."):

#         if isinstance(model, AC_Filter_Net):
#             losses, epoch_losses = f_train_rnn(model, losses, loss, opt, y_true, args)
#         else:
#             losses, epoch_losses = f_train(model, losses, loss, opt, y_true, args)
    
#         mean_loss = np.mean(epoch_losses)
#         mean_losses.append(mean_loss)

#         log_mean_loss = np.log(mean_loss)

#         if smoothed_log_mean_losses == []:
#             smoothed_log_mean_loss = log_mean_loss
#         else:
#             smoothed_log_mean_loss = smoothed_log_mean_losses[-1]
        
#         smoothed_log_mean_loss = f_running_avg(log_mean_loss, smoothed_log_mean_loss)
#         smoothed_log_mean_losses.append(smoothed_log_mean_loss)

#         epochs_no_improve, min_smoothed_log_mean_loss, trigger_prune = f_trigger_pruning(
#             smoothed_log_mean_loss, min_smoothed_log_mean_loss, epochs_no_improve, n_epochs_stop, epoch_dec
#         )

#         if smoothed_log_mean_loss <= best_smoothed_log_mean_loss:
#             best_smoothed_log_mean_loss = smoothed_log_mean_loss
#             model.best_weights = copy.deepcopy(model.state_dict())
#             model.best_epoch = epoch

#         if trigger_prune:
#             min_smoothed_log_mean_loss = np.Infinity
#             epochs_no_improve = 0 

#             if args.training_mode != 1: 
#                 f_plot_velocities(model, y_valid, args)
# #                 if isinstance(model, AC_Filter_Net):
# #                     f_plot_lookahead(model, y_valid, args, num_samples=3)
# #                 else:
# #                     f_plot_velocities(model, y_valid, args)
#                 f_plot_weights(model)

#                 prune_epochs.append(epoch)
#                 print(f"\n--- Pruning Triggered ({prune_mode}) at Epoch {epoch} ---")
#                 f_display_weights(model)

#                 for _ in np.arange(num_to_prune):
#                     num_pruned += 1

#                     # --- SWITCHABLE PRUNING LOGIC ---
#                     if prune_mode == 'gradient':
#                         layer_idx, weight_idx = f_prune_by_gradient(model, y_true, loss, args)
#                     else:
#                         layer_idx, weight_idx = f_prune_by_effect(model, y_valid, loss, args)

#                     if layer_idx == np.Infinity:
#                         break
                    
#                     linear_layers = f_get_linear_layers(model)
#                     linear_layers[layer_idx].compute_mask(weight_idx)

#                 print(f"Pruned edges: {num_pruned}")
#                 print(f"Remaining edges: {num_edges - num_pruned}")
#                 args.training_mode = 0 

#             print(f"Current Training Mode: {args.training_mode}")
#             print('----------------------------------------------')
#             opt = optim.Adam(model.parameters(), lr=lr)

#         if layer_idx == np.Infinity:
#             break

#     return losses, mean_losses, smoothed_log_mean_losses, prune_epochs

def f_train_model(y_true, y_valid, t, model, loss, opt, w, args, 
                  y_ground_truth=None, max_epochs=500, lr=2e-3, 
                  n_epochs_stop=4, epoch_dec=1, num_to_prune=1, 
                  prune_mode='gradient', use_norm=False):
    """
    Main training and pruning loop for Neural SINDy.
    y_true: Noisy training data.
    y_valid: Noisy validation data.
    y_ground_truth: Clean validation data (optional, used for plotting).
    """

    losses = []
    mean_losses = []
    smoothed_log_mean_losses = []

    num_edges = f_number_of_edges(model)
    layer_idx = 0
    trigger_prune = False
    epochs_no_improve = 0
    min_smoothed_log_mean_loss = np.Infinity
    best_smoothed_log_mean_loss = np.Infinity
    num_pruned = 0
    prune_epochs = []

    for epoch in tqdm(np.arange(max_epochs) + 1, desc="Training progress..."):

        if isinstance(model, (AC_Filter_Net, AC_Filter_Norm_Net, AC_Filter_PreNorm_Net)):
            losses, epoch_losses = f_train_rnn(model, losses, loss, opt, y_true, args, use_norm=use_norm)
        else:
            losses, epoch_losses = f_train(model, losses, loss, opt, y_true, args)
    
        mean_loss = np.mean(epoch_losses)
        mean_losses.append(mean_loss)

        log_mean_loss = np.log(mean_loss)

        if smoothed_log_mean_losses == []:
            smoothed_log_mean_loss = log_mean_loss
        else:
            smoothed_log_mean_loss = smoothed_log_mean_losses[-1]
        
        smoothed_log_mean_loss = f_running_avg(log_mean_loss, smoothed_log_mean_loss)
        smoothed_log_mean_losses.append(smoothed_log_mean_loss)

        epochs_no_improve, min_smoothed_log_mean_loss, trigger_prune = f_trigger_pruning(
            smoothed_log_mean_loss, min_smoothed_log_mean_loss, epochs_no_improve, n_epochs_stop, epoch_dec
        )

        if smoothed_log_mean_loss <= best_smoothed_log_mean_loss:
            best_smoothed_log_mean_loss = smoothed_log_mean_loss
            model.best_weights = copy.deepcopy(model.state_dict())
            model.best_epoch = epoch

        if trigger_prune:
            min_smoothed_log_mean_loss = np.Infinity
            epochs_no_improve = 0 

            if args.training_mode != 1: 
                # --- UPDATED PLOTTING CALL ---
                # Pass both the noisy validation and the optional clean ground truth
                f_plot_velocities(model, y_valid, args, y_ground_truth=y_ground_truth)
                
                f_plot_weights(model)
                
                # Plot log-mean loss per epoch with pruning markers
                f_plot(np.log(mean_losses), xlabel='Epoch', ylabel='Log Mean Loss', title='Log Mean Losses')
                # Plot smoothed log-mean loss per epoch
                f_plot(smoothed_log_mean_losses, color='b', alpha=0.5)
                if prune_epochs:
                    for x_e in prune_epochs:
                        plt.axvline(x=x_e, color='k', ls='--', alpha=0.5, label='Pruning Event')
                plt.show()

            prune_epochs.append(epoch)
            print(f"\n--- Pruning Triggered ({prune_mode}) at Epoch {epoch} ---")
            f_display_weights(model)

            for _ in np.arange(num_to_prune):
                num_pruned += 1

                # --- SWITCHABLE PRUNING LOGIC ---
                if prune_mode == 'gradient':
                    layer_idx, weight_idx = f_prune_by_gradient(model, y_true, loss, args)
                else:
                    layer_idx, weight_idx = f_prune_by_effect(model, y_valid, loss, args)

                if layer_idx == np.Infinity:
                    layer_idx = 0 # Reset to allow loop to continue if possible
                    break
                
                linear_layers = f_get_linear_layers(model)
                linear_layers[layer_idx].compute_mask(weight_idx)

            print(f"Pruned edges: {num_pruned}")
            print(f"Remaining edges: {num_edges - num_pruned}")
            args.training_mode = 0 

            print(f"Current Training Mode: {args.training_mode}")
            print('----------------------------------------------')
            opt = optim.Adam(model.parameters(), lr=lr)

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

################################################################################
################################################################################ 

# def f_prune_by_gradient(model, y_true, loss_fn, args):
#     """
#     Prunes the edge with the minimum |weight * gradient| score.
#     Scalability: O(1) backward pass vs O(N) forward passes.
#     """
#     model.train()
#     # 1. Get a representative batch of data
#     batch_y0, batch_t, batch_y = f_get_batch(model, y_true, 0, args)
    
#     # 2. Forward pass and compute loss
#     pred_y = f_get_prediction(model, batch_y0, batch_t, args)
#     loss = loss_fn(batch_y, pred_y)
    
#     # 3. Backward pass to populate .grad attributes
#     model.zero_grad()
#     loss.backward()

#     # 4. Calculate importance scores for all linear layers
#     linear_layers = f_get_linear_layers(model)
#     min_score = float('inf')
#     target_layer_idx = None
#     target_weight_idx = None

#     for l_idx, layer in enumerate(linear_layers):
#         if layer.weight.grad is None:
#             continue
            
#         # Importance = |weight * gradient|
#         # We only care about weights where the mask is currently 1.0
#         scores = torch.abs(layer.weight * layer.weight.grad)
#         masked_scores = torch.where(layer.mask > 0, scores, torch.tensor(float('inf'), device=args.device))
        
#         layer_min_val, layer_min_flat_idx = torch.min(masked_scores.view(-1), dim=0)
        
#         if layer_min_val < min_score:
#             min_score = layer_min_val
#             target_layer_idx = l_idx
#             # Convert flat index back to 2D tuple (row, col)
#             row = layer_min_flat_idx // layer.weight.shape[1]
#             col = layer_min_flat_idx % layer.weight.shape[1]
#             target_weight_idx = (row.item(), col.item())

#     # Return np.Infinity if no pruneable weights were found
#     if target_layer_idx is None or min_score == float('inf'):
#         return np.Infinity, np.Infinity

#     return target_layer_idx, target_weight_idx

def f_prune_by_gradient(model, y_true, loss_fn, args, num_batches=5):
    """
    Universal Gradient-based Pruning.
    Correctly aligns targets for both standard SINDy and Parallel RNN models.
    """
    model.train()
    linear_layers = f_get_linear_layers(model)
    importance_scores = [torch.zeros_like(l.weight) for l in linear_layers]
    
    h_len = getattr(args, 'num_t_steps', 10)
    f_len = getattr(args, 'batch_time', 4)

    for i in range(num_batches):
        model.zero_grad()
        
        # --- 1. HANDLE RNN / ATTENTION CASE ---
        if isinstance(model, (AC_Filter_Net, AC_Filter_Norm_Net, AC_Filter_PreNorm_Net)):
            batch_y0, _, batch_full = f_get_batch_rnn(y_true, i, args)
            
            # Forward: [Batch, 10, 8]
            pred_y = model(None, batch_y0) 
            
            # Target Alignment (Must match f_train_rnn logic)
            y_feats = batch_full[:, :, 1:] # Drop constant 1
            target_windows = y_feats.unfold(1, f_len, 1).transpose(2, 3)
            target_y = target_windows[:, 1:h_len+1, :, :].reshape(batch_full.shape[0], h_len, -1)
            
        # --- 2. HANDLE STANDARD SINDY CASE ---
        else:
            from .model_utils import f_get_batch
            batch_y0, batch_t, batch_y = f_get_batch(model, y_true, i, args)
            # Standard prediction (usually [Batch, Dim])
            pred_y = f_get_prediction(model, batch_y0, batch_t, args)
            target_y = batch_y

        # Calculate loss and backprop
        loss = loss_fn(pred_y, target_y)
        loss.backward()
        
        # Accumulate Sensitivity Score
        for l_idx, layer in enumerate(linear_layers):
            if layer.weight.grad is not None:
                importance_scores[l_idx] += torch.abs(layer.weight * layer.weight.grad)

    # --- WEIGHT SELECTION LOGIC ---
    min_score = float('inf')
    target_layer_idx = None
    target_weight_idx = None

    for l_idx, layer in enumerate(linear_layers):
        masked_scores = torch.where(layer.mask > 0, 
                                    importance_scores[l_idx], 
                                    torch.tensor(float('inf'), device=args.device))
        
        layer_min_val, layer_min_flat_idx = torch.min(masked_scores.view(-1), dim=0)
        
        if layer_min_val < min_score:
            min_score = layer_min_val
            target_layer_idx = l_idx
            num_cols = layer.weight.shape[1]
            row = layer_min_flat_idx // num_cols
            col = layer_min_flat_idx % num_cols
            target_weight_idx = (row.item(), col.item())

    if target_layer_idx is None or min_score == float('inf'):
        return np.Infinity, np.Infinity

    return target_layer_idx, target_weight_idx

