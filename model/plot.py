import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from .model_utils import f_get_linear_layers, f_get_Hadamard_layers
from .models import AC_Filter_Net, AC_Filter_Norm_Net, AC_Filter_PreNorm_Net

################################################################################
################################################################################

# def f_plot_velocities(model, y_true, args):
#     """Predict true and predicted velocities"""

#     with torch.no_grad():
#         v_pred = model.forward([],y_true[:-1]).detach().cpu().numpy().squeeze().T[1:] # Get predictions

#     v_euler = ((y_true[1:] - y_true[:-1])/args.dt).detach().cpu().numpy().squeeze().T[1:] # Euler approx of velocity

#     # Plot velocities in all dimensions
#     for i in np.arange(np.shape(v_euler)[0]):
#         fig = plt.figure()
#         plt.plot(v_euler[i])
#         plt.plot(v_pred[i],'k--')
#         plt.legend(["Ground Truth", "Prediction"], loc ="lower right")
#         plt.grid()
#         plt.show()

# def f_plot_velocities(model, y_valid, args, y_ground_truth=None):
#     """
#     Unified Velocity Plotter. 
#     Works for:
#     1. AC_Filter_Net (RNN/Attention multi-step windows)
#     2. Standard AC_Net/HLM_Net (One-step predictions)
#     """
#     model.eval()
#     h_len = getattr(args, 'num_t_steps', 10)
#     dt = getattr(args, 'dt', 0.01)
    
#     with torch.no_grad():
#         # --- CASE 1: AC_Filter_Net (RNN/Attention Windowed Case) ---
#         if isinstance(model, (AC_Filter_Net, AC_Filter_Norm_Net)):
#             inputs = y_valid.to(args.device)
#             if inputs.dim() == 4:
#                 inputs = inputs.squeeze(2)
            
#             # 1. Multi-step prediction
#             pred_full = model(None, inputs[:, :h_len, :])
            
#             # 2. Extract Velocity using the first predicted future point
#             # s1_pos is the denoised position of the LAST point in the history window
#             s1_pos = model.latent_state[:, -1, 1:3].cpu().numpy()
#             pred_pos = pred_full[:, -1, 0:2].cpu().numpy()
#             v_pred = ((pred_pos - s1_pos) / dt).T
            
#             # 3. Ground Truths (Observed Noisy vs Clean Unseen)
#             y_np = inputs.cpu().numpy()
#             v_noisy_euler = ((y_np[:, h_len, 1:] - y_np[:, h_len-1, 1:]) / dt).T
            
#             v_clean_euler = None
#             if y_ground_truth is not None:
#                 yc = y_ground_truth.to(args.device)
#                 if yc.dim() == 4: yc = yc.squeeze(2)
#                 yc_np = yc.cpu().numpy()
#                 v_clean_euler = ((yc_np[:, h_len, 1:] - yc_np[:, h_len-1, 1:]) / dt).T

#         # --- CASE 2: Standard SINDy / AC_Net (Single-step Prediction) ---
#         else:
#             inputs = y_valid.to(args.device)
#             # Standard models usually return velocity directly [Batch, Dim]
#             v_pred_raw = model.forward([], inputs[:-1]).detach().cpu().numpy().squeeze()
            
#             # Ensure shape is [Features, Time]
#             if v_pred_raw.ndim > 1:
#                 v_pred = v_pred_raw.T[1:] # Drop constant column 0
#             else:
#                 v_pred = v_pred_raw.reshape(1, -1)
            
#             # Euler Ground Truth from input
#             v_noisy_euler = ((inputs[1:] - inputs[:-1]) / dt).detach().cpu().numpy().squeeze().T[1:]
            
#             v_clean_euler = None
#             if y_ground_truth is not None:
#                 yc = y_ground_truth.to(args.device)
#                 v_clean_euler = ((yc[1:] - yc[:-1]) / dt).detach().cpu().numpy().squeeze().T[1:]

#     # --- Unified Plotting Logic ---
#     dims = ['X Velocity', 'Y Velocity']
#     # Iterate through the available features (X, Y)
#     for i in range(v_pred.shape[0]):
#         plt.figure(figsize=(10, 4))
        
#         # Plot Noisy Input (Blue Cloud)
#         plt.plot(v_noisy_euler[i], color='blue', alpha=0.15, label="Noisy Observed (Euler)")
        
#         # Plot Clean Truth (Green Line)
#         if v_clean_euler is not None:
#             plt.plot(v_clean_euler[i], color='green', alpha=0.6, label="Clean Ground Truth")
            
#         # Plot Model's Recovered Dynamics (Dashed Black Line)
#         plt.plot(v_pred[i], color='black', linestyle='--', linewidth=2, label="AC-Net Recovery")
        
#         plt.title(f"Dynamic Tracking: {dims[i] if i < 2 else f'Dim {i}'}")
#         plt.ylabel("Velocity")
#         plt.legend(loc='upper right')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()

# def f_plot_velocities(model, y_valid, args, y_ground_truth=None):
#     """
#     Unified Velocity Plotter. 
#     Handles both AC_Filter_Net and AC_Filter_Norm_Net without model changes.
#     """
#     model.eval()
#     h_len = getattr(args, 'num_t_steps', 10)
#     dt = getattr(args, 'dt', 0.01)
    
#     with torch.no_grad():
#         # --- CASE 1: AC_Filter_Net / AC_Filter_Norm_Net ---
#         if isinstance(model, (AC_Filter_Net, AC_Filter_Norm_Net)):
#             inputs = y_valid.to(args.device)
#             if inputs.dim() == 4:
#                 inputs = inputs.squeeze(2)
            
#             # 1. Multi-step prediction [Batch, Seq, 8]
#             pred_full = model(None, inputs[:, :h_len, :])
            
#             # 2. Extract Velocity using the original logic: (Next_Pos - Current_Pos) / dt
#             # Use the first predicted point (indices 0:2 of the 8 features)
#             pred_pos = pred_full[:, -1, 0:2].cpu().numpy()
#             s1_pos = model.latent_state[:, -1, 1:3].cpu().numpy()
#             v_pred = ((pred_pos - s1_pos) / dt).T # Shape [2, Batch]
            
#             # 3. Ground Truths (Observed Noisy vs Clean Unseen)
#             y_np = inputs.cpu().numpy()
#             v_noisy_euler = ((y_np[:, h_len, 1:] - y_np[:, h_len-1, 1:]) / dt).T
            
#             v_clean_euler = None
#             if y_ground_truth is not None:
#                 yc = y_ground_truth.to(args.device)
#                 if yc.dim() == 4: yc = yc.squeeze(2)
#                 yc_np = yc.cpu().numpy()
#                 v_clean_euler = ((yc_np[:, h_len, 1:] - yc_np[:, h_len-1, 1:]) / dt).T

#         # --- CASE 2: Standard SINDy / AC_Net ---
#         else:
#             inputs = y_valid.to(args.device)
#             v_pred_raw = model.forward([], inputs[:-1]).detach().cpu().numpy().squeeze()
            
#             if v_pred_raw.ndim > 1:
#                 v_pred = v_pred_raw.T[1:] 
#             else:
#                 v_pred = v_pred_raw.reshape(1, -1)
            
#             v_noisy_euler = ((inputs[1:] - inputs[:-1]) / dt).detach().cpu().numpy().squeeze().T[1:]
            
#             v_clean_euler = None
#             if y_ground_truth is not None:
#                 yc = y_ground_truth.to(args.device)
#                 v_clean_euler = ((yc[1:] - yc[:-1]) / dt).detach().cpu().numpy().squeeze().T[1:]

#     # --- Unified Plotting Logic ---
#     dims = ['X Velocity', 'Y Velocity']
#     for i in range(v_pred.shape[0]):
#         plt.figure(figsize=(10, 4))
        
#         # Plot Noisy Input (Blue Cloud)
#         plt.plot(v_noisy_euler[i], color='blue', alpha=0.15, label="Noisy Observed (Euler)")
        
#         # Plot Clean Truth (Green Line)
#         if v_clean_euler is not None:
#             plt.plot(v_clean_euler[i], color='green', alpha=0.6, label="Clean Ground Truth")
            
#         # Plot Model's Recovered Dynamics (Dashed Black Line)
#         plt.plot(v_pred[i], color='black', linestyle='--', linewidth=2, label="AC-Net Recovery")
        
#         plt.title(f"Dynamic Tracking: {dims[i] if i < 2 else f'Dim {i}'}")
#         plt.ylabel("Velocity")
#         plt.legend(loc='upper right')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()

def f_plot_velocities(model, y_valid, args, y_ground_truth=None):
    """
    Unified Velocity Plotter.
    Both AC_Filter_Net and AC_Filter_Norm_Net now output physical-scale values,
    so no rescaling is needed here — the two filter cases are identical.
    """
    model.eval()
    h_len = getattr(args, 'num_t_steps', 10)
    dt = getattr(args, 'dt', 0.01)

    with torch.no_grad():
        # --- CASE 1: AC_Filter_Net / AC_Filter_Norm_Net ---
        if isinstance(model, (AC_Filter_Net, AC_Filter_Norm_Net, AC_Filter_PreNorm_Net)):
            inputs = y_valid.to(args.device)
            if inputs.dim() == 4:
                inputs = inputs.squeeze(2)

            # 1. Multi-step prediction — always physical scale
            pred_full = model(None, inputs[:, :h_len, :])

            # 2. Extract velocity: (Next_Pos - Current_Pos) / dt
            # Both pred_pos and s1_pos are now in physical scale
            pred_pos = pred_full[:, -1, 0:2].cpu().numpy()
            s1_pos   = model.latent_state[:, -1, 1:3].cpu().numpy()
            v_pred   = ((pred_pos - s1_pos) / dt).T  # [2, Batch]

            # 3. Ground truths (physical scale)
            y_np = inputs.cpu().numpy()
            v_noisy_euler = ((y_np[:, h_len, 1:] - y_np[:, h_len-1, 1:]) / dt).T

            v_clean_euler = None
            if y_ground_truth is not None:
                yc = y_ground_truth.to(args.device)
                if yc.dim() == 4:
                    yc = yc.squeeze(2)
                yc_np = yc.cpu().numpy()
                v_clean_euler = ((yc_np[:, h_len, 1:] - yc_np[:, h_len-1, 1:]) / dt).T

        # --- CASE 2: Standard SINDy / AC_Net (single-step) ---
        else:
            inputs = y_valid.to(args.device)
            v_pred_raw = model.forward([], inputs[:-1]).detach().cpu().numpy().squeeze()
            if v_pred_raw.ndim > 1:
                v_pred = v_pred_raw.T[1:]
            else:
                v_pred = v_pred_raw.reshape(1, -1)

            v_noisy_euler = ((inputs[1:] - inputs[:-1]) / dt).detach().cpu().numpy().squeeze().T[1:]

            v_clean_euler = None
            if y_ground_truth is not None:
                yc = y_ground_truth.to(args.device)
                v_clean_euler = ((yc[1:] - yc[:-1]) / dt).detach().cpu().numpy().squeeze().T[1:]

    # --- Unified Plotting Logic ---
    dims = ['X Velocity', 'Y Velocity']
    for i in range(v_pred.shape[0]):
        plt.figure(figsize=(10, 4))
        plt.plot(v_noisy_euler[i], color='blue', alpha=0.15, label="Noisy Observed (Euler)")
        if v_clean_euler is not None:
            plt.plot(v_clean_euler[i], color='green', alpha=0.6, label="Clean Ground Truth")
        plt.plot(v_pred[i], color='black', linestyle='--', linewidth=2, label="AC-Net Recovery")
        plt.title(f"Dynamic Tracking: {dims[i] if i < 2 else f'Dim {i}'}")
        plt.ylabel("Velocity")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

################################################################################
################################################################################
    
def f_plot_lookahead(model, y_true_windows, args, num_samples=3):
    """
    Plots the 4-step future prediction for a few sample windows.
    y_true_windows: [Batch, Hist+Future, 1, Dim]
    """
    model.eval()
    h_len = args.num_t_steps
    f_len = args.batch_time # 4
    
    # 1. Pick sample windows (e.g., start, middle, end)
    indices = np.linspace(0, len(y_true_windows)-1, num_samples, dtype=int)
    
    for idx in indices:
        window = y_true_windows[idx:idx+1].to(args.device).squeeze(2) # [1, Hist+Future, Dim]
        
        with torch.no_grad():
            # Get the 4-step prediction [1, 8]
            pred_seq = model(None, window[:, :h_len, :]).cpu().numpy().reshape(f_len, 2)
        
        # Ground Truth
        truth_seq = window[0, h_len:, 1:].cpu().numpy() # [4, 2]
        history = window[0, :h_len, 1:].cpu().numpy()   # [10, 2]
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        dims = ['x velocity', 'y velocity']
        
        for i in range(2):
            # Plot History
            axes[i].plot(np.arange(h_len), history[:, i], 'g-', label='History (Noisy)')
            # Plot Truth
            axes[i].plot(np.arange(h_len, h_len+f_len), truth_seq[:, i], 'b-', label='Ground Truth')
            # Plot Prediction
            axes[i].plot(np.arange(h_len, h_len+f_len), pred_seq[:, i], 'r--', label='AC Prediction')
            
            axes[i].set_title(dims[i])
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
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

def f_plot(input,xlabel=None,ylabel=None,title=None,color="black", alpha=1.0):
    """General plotting function"""
    plt.plot(input, '-', color=color, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    # plt.show()

################################################################################
################################################################################

