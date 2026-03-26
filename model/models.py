import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import MaskedLinearLayer, MaskedLinearLayer2, MultiplicativeLayer, MaskedHadamardLinearLayer
from .model_utils import f_get_linear_layers
from .normalization import FeatureNorm

################################################################################
# AC_Net3: Standard 3-channel Arithmetic Circuit
################################################################################

class AC_Net3(torch.nn.Module):
    """
    Arithmetic Circuit Network (3-Channel).
    A modular architecture that alternates between masked linear layers for feature 
    transformation and multiplicative layers for nonlinear interaction discovery.
    """
    def __init__(self, num_in, num_out, w1):
        super(AC_Net3, self).__init__()
        self.h1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.h2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.h3 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.m1 = MultiplicativeLayer()
        self.m2 = MultiplicativeLayer()
        self.m3 = MultiplicativeLayer()

    def forward(self, t, inputs):
        a1 = self.h1(inputs) 
        a2 = self.m1(a1)     
        a3 = self.h2(inputs) 
        a4 = self.m2(a3)     
        a5 = self.h3(inputs) 
        a6 = self.m3(a5)     
        return torch.hstack((a2, a4, a6)).to(self.h1.weight.device)

    def initialize_const(self, init_array, num_in, num_out):
        for param in self.parameters():
            param.data = nn.parameter.Parameter(init_array)

################################################################################
# AC_Net_2D: 2D State-Space Identification
################################################################################

class AC_Net_2D(torch.nn.Module):
    """
    Arithmetic Circuit Network (2-Channel).
    Specifically designed for identifying 2D dynamical systems. Includes 
    standardized attributes for tracking best model performance during training.
    """
    def __init__(self, num_in, num_out, w1):
        super(AC_Net_2D, self).__init__()
        self.h1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.h2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.m1 = MultiplicativeLayer()
        self.m2 = MultiplicativeLayer()
        self.best_model = {}
        self.best_epoch = 0
        self.prev_std_dev = 1
        self.std_dev = 1

    def forward(self, t, inputs):
        a1 = self.h1(inputs) 
        a2 = self.m1(a1) 
        a3 = self.h2(inputs) 
        a4 = self.m2(a3) 
        a5 = torch.zeros_like(a2) 
        return torch.cat((a5, a2, a4), 2).to(self.h1.weight.device)

    def initialize_const(self, init_array, num_in, num_out):
        for param in self.parameters():
            param.data = nn.parameter.Parameter(init_array)

#     def initialize_rand_normal(self, mean=0, std=0.1):
#         layers = f_get_linear_layers(self)
#         for layer in layers:
#             nn.init.normal_(layer.weight, mean=mean, std=std)
#             nn.init.constant_(layer.bias, val=0)

################################################################################
# AC_Net_3D: 3D State-Space Identification
################################################################################

class AC_Net_3D(torch.nn.Module):
    """
    3D Arithmetic Circuit with Residuals.
    """
    def __init__(self, num_in, num_out, w1):
        super(AC_Net_3D, self).__init__()
        # ALL input layers MUST use 'num_in'
        self.h1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.h2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.h3 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        
        # Residual layers also MUST use 'num_in'
        self.h4 = MaskedLinearLayer(size_in=num_in, size_out=1)
        self.h5 = MaskedLinearLayer(size_in=num_in, size_out=1)
        self.h6 = MaskedLinearLayer(size_in=num_in, size_out=1)
        
        self.m1 = MultiplicativeLayer()
        self.best_model = {}
        self.best_epoch = 0
        self.std_dev = 1.0 # Added to prevent AttributeError during weights display

    def forward(self, t, inputs):
        a1 = self.h1(inputs) 
        a2 = self.m1(a1) 
        a3 = self.h2(inputs) 
        a4 = self.m1(a3) 
        a5 = self.h3(inputs) 
        a6 = self.m1(a5) 
        
        # Branch assembly
        a7 = torch.zeros_like(a2) # Column 0
        r1 = self.h4(inputs)      # For Column 1
        r2 = self.h5(inputs)      # For Column 2
        r3 = self.h6(inputs)      # For Column 3

        # Final concatenation: Resulting shape [Batch, Samples, 4] 
        # (One zero column + 3 state derivative columns)
        return torch.cat((a7, a2 + r1, a4 + r2, a6 + r3), 2).to(self.h1.weight.device)

################################################################################
# Sin_AC_Net: periodic Basis Identification
################################################################################

class Sin_AC_Net_2(torch.nn.Module):
    """
    Second-Order Sine-AC Network.
    Integrates sinusoidal activation functions into the arithmetic circuit 
    structure to facilitate the discovery of periodic dynamical components.
    """
    def __init__(self, num_in, num_out, W):
        super(Sin_AC_Net_2, self).__init__()
        self.h1 = MaskedLinearLayer(size_in=num_in, size_out=W[0])
        self.m1 = MultiplicativeLayer()
        self.h2 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.h3 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.h4 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.h7 = MaskedLinearLayer(size_in=W[1], size_out=W[1])
        self.h8 = MaskedLinearLayer(size_in=W[1], size_out=W[1])
        self.h9 = MaskedLinearLayer(size_in=W[1], size_out=W[1])
        self.best_weights = {}
        self.best_epoch = 0

    def forward(self, t, inputs):
        a1 = self.h1(inputs)
        a2 = torch.sin(self.h2(inputs))
        a3 = torch.sin(self.h3(inputs))
        a4 = torch.sin(self.h4(inputs))
        a7 = self.h7(a2)
        a8 = self.h8(a3)
        a9 = self.h9(a4)
        a12 = self.m1(a1)
        return (a12 + a7 + a8 + a9).to(self.h1.weight.device)

    def initialize_const(self, init_array, num_in, num_out):
        for param in self.parameters():
            param.data = nn.parameter.Parameter(init_array)

################################################################################
# Sin_AC_Net_4: High-Order periodic Identification
################################################################################

class Sin_AC_Net_4(torch.nn.Module):
    """
    Depth-4 Sine-Product Network.
    A deep architecture incorporating multiple stages of sinusoidal basis functions 
    and multiplicative layers to identify high-order (up to 4th order) terms.
    """
    def __init__(self, num_in, num_out, W):
        super(Sin_AC_Net_4, self).__init__()
        self.hA0_0 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.m0 = MultiplicativeLayer()
        self.hB0_0 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.hB1_0 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.hB2_0 = MaskedLinearLayer(size_in=num_in, size_out=W[1])
        self.hB0_1 = MaskedLinearLayer(size_in=1, size_out=1)
        self.hB1_1 = MaskedLinearLayer(size_in=1, size_out=1)
        self.hB2_1 = MaskedLinearLayer(size_in=1, size_out=1)
        self.hC0_0 = MaskedLinearLayer(size_in=num_in, size_out=W[0])
        self.m1 = MultiplicativeLayer()
        self.hD0_0 = MaskedLinearLayer(size_in=num_in, size_out=W[0])
        self.m2 = MultiplicativeLayer()
        self.hE0_0 = MaskedLinearLayer(size_in=num_in, size_out=W[0])
        self.m3 = MultiplicativeLayer()
        self.hF0_1 = MaskedLinearLayer(size_in=W[0], size_out=W[1])
        self.m4 = MultiplicativeLayer()
        self.m5 = MultiplicativeLayer()
        self.hG0_0 = MaskedLinearLayer(size_in=num_in, size_out=W[2])
        self.m6 = MultiplicativeLayer()

    def forward(self, t, inputs):
        device = inputs.device
        aA0_0 = self.hA0_0(inputs) 
        aB0_0 = torch.sin(self.hB0_0(inputs))
        aB1_0 = torch.sin(self.hB1_0(inputs))
        aB2_0 = torch.sin(self.hB2_0(inputs))
        aC0_0 = self.hC0_0(inputs) 
        aC0_1 = self.m1(aC0_0) 
        aC0_2 = torch.sin(aC0_1)
        aD0_0 = self.hD0_0(inputs) 
        aD0_1 = self.m2(aD0_0) 
        aD0_2 = torch.sin(aD0_1)
        aE0_0 = self.hE0_0(inputs) 
        aE0_1 = self.m3(aE0_0) 
        aC0_2_aD0_2 = torch.hstack((aC0_2, aD0_2))
        aF0_1 = self.hF0_1(aC0_2_aD0_2) 
        aG0_0 = self.hG0_0(inputs) 
        aO1 = self.m0(aA0_0) 
        aO2 = self.hB0_1(aB0_0)
        aO3 = self.hB1_1(aB1_0)
        aO4 = self.hB2_1(aB2_0)
        aO5 = aF0_1
        aO6 = self.m4(torch.hstack((aO1, aF0_1)))
        aO7 = self.m5(aC0_2_aD0_2)
        aO8 = torch.sin(aF0_1)
        aO9 = self.m6(aG0_0)
        output = aO1 + aO2 + aO3 + aO4 + aO5 + aO6 + aO7 + aO8 + aO9
        return output.to(device)

    def initialize_const(self, init_array, num_in, num_out):
        for param in self.parameters():
            param.data = nn.parameter.Parameter(init_array)

################################################################################
# HLM_Net_2D: Hadamard-Linear-Multiply Architecture
################################################################################

class HLM_Net_2D(torch.nn.Module):
    """
    Hadamard-Linear-Multiply Network (2D).
    Utilizes Hadamard (element-wise) linear layers to induce sparsity at the 
    feature level before multiplicative combination.
    """
    def __init__(self, num_in, num_out, w1):
        super(HLM_Net_2D, self).__init__()
        self.h1 = MaskedHadamardLinearLayer(size_in=num_in, size_out=w1)
        self.h2 = MaskedHadamardLinearLayer(size_in=num_in, size_out=w1)
        self.m1 = MultiplicativeLayer()
        self.m2 = MultiplicativeLayer()
        self.best_model = {}
        self.best_epoch = 0

    def forward(self, t, inputs):
        a1 = self.h1(inputs)
        a2 = self.m1(a1)
        b1 = self.h2(inputs)
        b2 = self.m2(b1)
        c1 = torch.zeros_like(a2) 
        return torch.cat((c1, a2, b2), 2).to(self.h1.weight.device)

    def initialize_const(self, init_array, num_in, num_out):
        for param in self.parameters():
            param.data = nn.parameter.Parameter(init_array)

#     def initialize_rand_normal(self, mean=0, std=0.1):
#         layers = f_get_linear_layers(self)
#         for layer in layers:
#             nn.init.normal_(layer.weight, mean=mean, std=std)
#             nn.init.constant_(layer.bias, val=0)

################################################################################
# HLMSD_Net_2D: Rational and period System Discovery
################################################################################

class HLMSD_Net_2D(torch.nn.Module):
    """
    Hadamard-Linear-Multiply-Sine-Divide (HLMSD) Network.
    A comprehensive architecture for rational dynamical systems. Capable of 
    representing division and periodic terms through dedicated branches.
    """
    def __init__(self, num_in, num_out, w1):
        super(HLMSD_Net_2D, self).__init__()
        self.ha1 = MaskedLinearLayer2(size_in=num_in, size_out=w1)
        self.hb1 = MaskedLinearLayer2(size_in=num_in, size_out=w1)
        self.m = MultiplicativeLayer()
        self.ha2 = MaskedLinearLayer2(size_in=num_in, size_out=1)
        self.ha3 = MaskedLinearLayer2(size_in=1, size_out=1)
        self.ha4 = MaskedLinearLayer2(size_in=1, size_out=1)
        self.hb2 = MaskedLinearLayer2(size_in=num_in, size_out=1)
        self.hb3 = MaskedLinearLayer2(size_in=1, size_out=1)
        self.hb4 = MaskedLinearLayer2(size_in=1, size_out=1)
        self.hb5 = MaskedLinearLayer2(size_in=num_in, size_out=1)
        self.hb6 = MaskedLinearLayer2(size_in=1, size_out=1)
        self.hb7 = MaskedLinearLayer2(size_in=1, size_out=1)
        self.hb8 = MaskedLinearLayer2(size_in=num_in, size_out=1)
        self.best_model = {}
        self.best_epoch = 0
        self.std_dev = 1.0

    def forward(self, t, inputs):
        device = inputs.device
        eps_val = 1e-2
        one = torch.tensor([[[1.]]], device=device)
        eps = torch.tensor([[[eps_val]]], device=device)
        a1 = self.ha1(inputs)
        a2 = self.m(a1)
        a3 = self.ha2(inputs)
        a4 = torch.sin(a3)
        a5 = torch.abs(self.ha3(one)) + one + eps + a4
        a6 = one / a5
        a7 = self.ha4(a6)
        b1 = self.hb1(inputs)
        b2 = self.m(b1)
        b3 = self.hb2(inputs)
        b4 = torch.sin(b3)
        b5 = torch.abs(self.hb3(one)) + one + eps + b4
        b6 = one / b5
        b7 = self.hb4(b6)
        b8 = self.hb5(inputs)
        b9 = torch.sin(b8)
        b10 = torch.abs(self.hb6(one)) + one + eps + b9
        b11 = one / b10
        b12 = self.hb7(b11)
        b13 = self.hb8(inputs)
        b14 = b7 * b13
        b15 = b12 * b13
        c1 = torch.zeros_like(a2).to(device) 
        return torch.cat((c1, a2 + a7, b2 + b14 + b15), dim=2)

    def initialize_const(self, init_array, num_in, num_out):
        for param in self.parameters():
            param.data = nn.parameter.Parameter(init_array)

#     def initialize_rand_normal(self, mean=0, std=0.1):
#         layers = f_get_linear_layers(self)
#         for layer in layers:
#             nn.init.normal_(layer.weight, mean=mean, std=std)
#             nn.init.constant_(layer.bias, val=0)
            
################################################################################
################################################################################

class AC_Filter_Net(nn.Module):
    def __init__(self, num_in, num_out, w1, num_t_steps=10, dt=0.01):
        super(AC_Filter_Net, self).__init__()
        self.num_t_steps = num_t_steps
        self.dt = dt

        self.f1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.f2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.g1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.g2 = MaskedLinearLayer(size_in=num_in, size_out=w1)

        self.m1 = MultiplicativeLayer()
        self.m2 = MultiplicativeLayer()

        self.mha1 = nn.MultiheadAttention(num_in, num_heads=1, batch_first=True)
        
        self.register_buffer('mask_v1', torch.tensor((0, 1, 1), dtype=torch.float32))
        self.register_buffer('mask_v2', torch.tensor((1, 0, 0), dtype=torch.float32))

        self.latent_state = None
        self.latent_v = None

    def _compute_velocity(self, state):
        v_out_1 = self.m1(self.f1(state))
        v_out_2 = self.m2(self.f2(state))
        v_state_1 = self.m1(self.g1(state))
        v_state_2 = self.m2(self.g2(state))
        return v_out_1, v_out_2, v_state_1, v_state_2

    def _get_next_state(self, current_state, v_state_1, v_state_2):
        z = torch.zeros_like(v_state_1) 
        delta = torch.cat((z, v_state_1, v_state_2), dim=2) * self.dt
        next_state = current_state + delta
        next_state[:, :, 0] = 1.0 # Maintain bias column
        return next_state

    def forward(self, t, inputs):
        L = inputs.size(1) 
        attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=inputs.device), diagonal=1)
        attn_out, _ = self.mha1(inputs, inputs, inputs, attn_mask=attn_mask, is_causal=True)

        # S1: [Batch, L, 3] -> (1, x, y)
        S1 = (attn_out * self.mask_v1) + self.mask_v2
        self.latent_state = S1 

        # Step rollout
        va1, va2, sa1, sa2 = self._compute_velocity(S1)
        S2 = self._get_next_state(S1, sa1, sa2) # S2 is now (1, x+dx, y+dy)

        vb1, vb2, sb1, sb2 = self._compute_velocity(S2)
        S3 = self._get_next_state(S2, sb1, sb2)

        vc1, vc2, sc1, sc2 = self._compute_velocity(S3)
        S4 = self._get_next_state(S3, sc1, sc2)

        vd1, vd2, _, _ = self._compute_velocity(S4)

        # CRITICAL: We only want to output the STATE dimensions (1 and 2), not the bias (0)
        # We output them as pairs for each step: [x_t+1, y_t+1, x_t+2, y_t+2, ...]
        out1 = S2[:, :, 1:3] # This IS S1 + v1*dt
        out2 = S3[:, :, 1:3] # This IS S2 + v2*dt
        out3 = S4[:, :, 1:3]
        
        # We need a 4th step to complete the 8-feature output
        # va_final = self._compute_velocity(S4)
        out4 = S4[:, :, 1:3] + torch.cat((vd1, vd2), dim=2) * self.dt

        return torch.cat((out1, out2, out3, out4), dim=2)
    
################################################################################
################################################################################

# class AC_Filter_Norm_Net(nn.Module):
#     def __init__(self, num_in, num_out, w1, num_t_steps=10, dt=0.01):
#         super(AC_Filter_Norm_Net, self).__init__()
#         self.num_t_steps = num_t_steps
#         self.dt = dt

#         # Mandatory Normalization for gradient health
#         self.norm = FeatureNorm(num_features=num_in)

#         # Arithmetic Circuit Layers
#         self.f1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
#         self.f2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
#         self.g1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
#         self.g2 = MaskedLinearLayer(size_in=num_in, size_out=w1)

#         self.m1 = MultiplicativeLayer()
#         self.m2 = MultiplicativeLayer()

#         self.mha1 = nn.MultiheadAttention(num_in, num_heads=1, batch_first=True)
        
#         self.register_buffer('mask_v1', torch.tensor((0, 1, 1), dtype=torch.float32))
#         self.register_buffer('mask_v2', torch.tensor((1, 0, 0), dtype=torch.float32))

#         self.best_model = {}
#         self.best_epoch = 0
#         self.latent_state = None
#         self.latent_v = None

#     def _compute_velocity(self, state):
#         v_out_1 = self.m1(self.f1(state))
#         v_out_2 = self.m2(self.f2(state))
#         v_state_1 = self.m1(self.g1(state))
#         v_state_2 = self.m2(self.g2(state))
#         return v_out_1, v_out_2, v_state_1, v_state_2

#     def _get_next_state(self, current_state, v_state_1, v_state_2):
#         z = torch.zeros_like(v_state_1) 
#         delta = torch.cat((z, v_state_1, v_state_2), dim=2) * self.dt
#         next_state = current_state + delta
#         next_state[:, :, 0] = 1.0 
#         return next_state

#     def forward(self, t, inputs):
#         L = inputs.size(1)
#         attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=inputs.device), diagonal=1)

#         # 1. Attention Denoising
#         attn_out, _ = self.mha1(inputs, inputs, inputs, attn_mask=attn_mask, is_causal=True)
#         S1_raw = (attn_out * self.mask_v1) + self.mask_v2
        
#         # 2. Normalization Setup
#         sigma = self.norm.running_sigma.detach() # [3] -> [sig_const, sig_x, sig_y]
#         sigma_xy = sigma[1:3]                   # [2] -> [sig_x, sig_y]
        
#         # 3. Enter the Unit World
#         S1 = self.norm(S1_raw)
#         was_training = self.norm.training
#         self.norm.eval() 

#         # 4. Recursive Rollout (Physics in Unit Space)
#         va1, va2, sa1, sa2 = self._compute_velocity(S1)
#         S2 = self._get_next_state(S1, sa1, sa2)
#         vb1, vb2, sb1, sb2 = self._compute_velocity(S2)
#         S3 = self._get_next_state(S2, sb1, sb2)
#         vc1, vc2, sc1, sc2 = self._compute_velocity(S3)
#         S4 = self._get_next_state(S3, sc1, sc2)
#         vd1, vd2, _, _ = self._compute_velocity(S4)

#         if was_training: self.norm.train()

#         # --- THE CRITICAL FIX FOR THE INDEX ERROR ---
#         # Plotters usually loop through the last dimension.
#         # S1 has 3 cols [1, x, y]. Latent_v MUST have 2 cols [vx, vy].
        
#         self.latent_state = S1 * (sigma + self.norm.eps)
        
#         # va1 and va2 are [Batch, L, 1]. Cat them to get [Batch, L, 2].
#         # Scale by sigma_xy to get real units.
#         self.latent_v = torch.cat((va1, va2), dim=2) * (sigma_xy + self.norm.eps)

#         # 5. Output Construction (8 features: 4 steps of x,y)
#         out1 = S2[:, :, 1:3]
#         out2 = S3[:, :, 1:3]
#         out3 = S4[:, :, 1:3]
#         out4 = S4[:, :, 1:3] + torch.cat((vd1, vd2), dim=2) * self.dt
        
#         output_cat = torch.cat((out1, out2, out3, out4), dim=2)
        
#         # 6. Final Scale back to Real World for Loss Function
#         # Reshape sigma_stack to [1, 1, 8] to ensure correct broadcasting
#         sigma_stack = sigma_xy.repeat(4).view(1, 1, 8).to(inputs.device)
        
#         return output_cat * (sigma_stack + self.norm.eps)


class AC_Filter_Norm_Net(nn.Module):
    def __init__(self, num_in, num_out, w1, num_t_steps=10, dt=0.01):
        super(AC_Filter_Norm_Net, self).__init__()
        self.num_t_steps = num_t_steps
        self.dt = dt
        # num_features=2: sigma tracks x and y only, never the bias column
        self.norm = FeatureNorm(num_features=2)

        self.f1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.f2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.g1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.g2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.m1 = MultiplicativeLayer()
        self.m2 = MultiplicativeLayer()
        self.mha1 = nn.MultiheadAttention(num_in, num_heads=1, batch_first=True)

        self.register_buffer('mask_v1', torch.tensor((0, 1, 1), dtype=torch.float32))
        self.register_buffer('mask_v2', torch.tensor((1, 0, 0), dtype=torch.float32))

        self.latent_state = None  # physical scale, for plotting
        self.latent_v = None      # physical scale, for plotting

    def _compute_velocity(self, state_norm):
        v_out_1   = self.m1(self.f1(state_norm))
        v_out_2   = self.m2(self.f2(state_norm))
        v_state_1 = self.m1(self.g1(state_norm))
        v_state_2 = self.m2(self.g2(state_norm))
        return v_out_1, v_out_2, v_state_1, v_state_2

    def _get_next_state(self, current_state_n, v_state_n1, v_state_n2):
        z = torch.zeros_like(v_state_n1)
        delta = torch.cat((z, v_state_n1, v_state_n2), dim=2) * self.dt
        next_state_n = current_state_n + delta
        next_state_n[:, :, 0] = 1.0  # restore bias column
        return next_state_n

    def _to_physical(self, state_n, sigma):
        """Convert a unit-world state [Batch, L, 3] back to physical scale."""
        out = state_n.clone()
        out[..., 1:] = state_n[..., 1:] * (sigma + self.norm.eps)
        return out

    def forward(self, t, inputs):
        L = inputs.size(1)
        attn_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=inputs.device), diagonal=1
        )

        # 1. Denoise via causal self-attention
        attn_out, _ = self.mha1(inputs, inputs, inputs,
                                 attn_mask=attn_mask, is_causal=True)
        S1_raw = (attn_out * self.mask_v1) + self.mask_v2
        # S1_raw: [Batch, L, 3] — cols [1.0, x, y] in physical scale

        # 2. Update sigma from state cols only, stopgrad
        if self.training:
            self.norm.update_stats(S1_raw[..., 1:])  # [Batch, L, 2]
        sigma = self.norm.running_sigma.detach()      # [2]

        # 3. Normalize state cols only; bias col stays exactly 1.0
        S1_n = S1_raw.clone()
        S1_n[..., 1:] = S1_raw[..., 1:] / (sigma + self.norm.eps)

        # 4. Rollout entirely in unit world
        va1, va2, sa1, sa2 = self._compute_velocity(S1_n)
        S2_n = self._get_next_state(S1_n, sa1, sa2)

        vb1, vb2, sb1, sb2 = self._compute_velocity(S2_n)
        S3_n = self._get_next_state(S2_n, sb1, sb2)

        vc1, vc2, sc1, sc2 = self._compute_velocity(S3_n)
        S4_n = self._get_next_state(S3_n, sc1, sc2)

        vd1, vd2, _, _ = self._compute_velocity(S4_n)

        # 5. Store physical-scale quantities for plotting — convert once here
        self.latent_state = self._to_physical(S1_n, sigma)
        self.latent_v = torch.cat((va1, va2), dim=2) * (sigma + self.norm.eps)

        # 6. Scale rollout outputs back to physical world before returning
        #    so that loss, plotting, and pruning all see physical-scale values
        #    and need no knowledge of normalization
        sigma_stack = sigma.repeat(4).view(1, 1, -1)  # [1, 1, 8]

        out1 = S2_n[..., 1:3] * (sigma + self.norm.eps)
        out2 = S3_n[..., 1:3] * (sigma + self.norm.eps)
        out3 = S4_n[..., 1:3] * (sigma + self.norm.eps)
        out4 = (S4_n[..., 1:3] + torch.cat((vd1, vd2), dim=2) * self.dt) \
               * (sigma + self.norm.eps)

        return torch.cat((out1, out2, out3, out4), dim=2)
        # output: [Batch, L, 8], physical scale — identical contract to AC_Filter_Net
            
################################################################################
################################################################################            
        
class AC_Filter_PreNorm_Net(nn.Module):
    def __init__(self, num_in, num_out, w1, num_t_steps=10, dt=0.01):
        super(AC_Filter_PreNorm_Net, self).__init__()
        self.num_t_steps = num_t_steps
        self.dt = dt
        # num_features=2: sigma tracks x and y only, never the bias column
        self.norm = FeatureNorm(num_features=2)

        self.f1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.f2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.g1 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.g2 = MaskedLinearLayer(size_in=num_in, size_out=w1)
        self.m1 = MultiplicativeLayer()
        self.m2 = MultiplicativeLayer()
        self.mha1 = nn.MultiheadAttention(num_in, num_heads=1, batch_first=True)

        self.register_buffer('mask_v1', torch.tensor((0, 1, 1), dtype=torch.float32))
        self.register_buffer('mask_v2', torch.tensor((1, 0, 0), dtype=torch.float32))

        self.latent_state = None  # physical scale, for plotting
        self.latent_v = None      # physical scale, for plotting

    def _compute_velocity(self, state_norm):
        v_out_1   = self.m1(self.f1(state_norm))
        v_out_2   = self.m2(self.f2(state_norm))
        v_state_1 = self.m1(self.g1(state_norm))
        v_state_2 = self.m2(self.g2(state_norm))
        return v_out_1, v_out_2, v_state_1, v_state_2

    def _get_next_state(self, current_state_n, v_state_n1, v_state_n2):
        z = torch.zeros_like(v_state_n1)
        delta = torch.cat((z, v_state_n1, v_state_n2), dim=2) * self.dt
        next_state_n = current_state_n + delta
        next_state_n[:, :, 0] = 1.0  # restore bias column
        return next_state_n

    def _to_physical(self, state_n, sigma):
        """Convert unit-world state [Batch, L, 3] back to physical scale."""
        out = state_n.clone()
        out[..., 1:] = state_n[..., 1:] * (sigma + self.norm.eps)
        return out

    def forward(self, t, inputs):
        # inputs: [Batch, L, 3] — cols [1.0, x, y] in physical scale
        L = inputs.size(1)
        attn_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=inputs.device), diagonal=1
        )

        # 1. Update sigma from raw state cols, stopgrad
        if self.training:
            self.norm.update_stats(inputs[..., 1:])  # [Batch, L, 2]
        sigma = self.norm.running_sigma.detach()      # [2]

        # 2. Normalize state cols before attention; bias col stays 1.0
        inputs_n = inputs.clone()
        inputs_n[..., 1:] = inputs[..., 1:] / (sigma + self.norm.eps)

        # 3. Causal self-attention on normalized inputs
        attn_out, _ = self.mha1(inputs_n, inputs_n, inputs_n,
                                 attn_mask=attn_mask, is_causal=True)

        # 4. Mask: keep only state cols from attention, restore bias col
        #    attn_out is already unit-scale so no further normalization needed
        S1_n = (attn_out * self.mask_v1) + self.mask_v2
        # S1_n: [Batch, L, 3] — cols [1.0, x_n, y_n], unit world

        # 5. Rollout entirely in unit world
        va1, va2, sa1, sa2 = self._compute_velocity(S1_n)
        S2_n = self._get_next_state(S1_n, sa1, sa2)

        vb1, vb2, sb1, sb2 = self._compute_velocity(S2_n)
        S3_n = self._get_next_state(S2_n, sb1, sb2)

        vc1, vc2, sc1, sc2 = self._compute_velocity(S3_n)
        S4_n = self._get_next_state(S3_n, sc1, sc2)

        vd1, vd2, _, _ = self._compute_velocity(S4_n)

        # 6. Store physical-scale quantities for plotting
        self.latent_state = self._to_physical(S1_n, sigma)
        self.latent_v = torch.cat((va1, va2), dim=2) * (sigma + self.norm.eps)

        # 7. Denormalize outputs to physical scale before returning
        out1 = S2_n[..., 1:3] * (sigma + self.norm.eps)
        out2 = S3_n[..., 1:3] * (sigma + self.norm.eps)
        out3 = S4_n[..., 1:3] * (sigma + self.norm.eps)
        out4 = (S4_n[..., 1:3] + torch.cat((vd1, vd2), dim=2) * self.dt) \
               * (sigma + self.norm.eps)

        return torch.cat((out1, out2, out3, out4), dim=2)
        # output: [Batch, L, 8], physical scale

################################################################################
################################################################################         

class AC_Filter_PreNorm_Net_ND(nn.Module):
    """
    N-dimensional generalization of AC_Filter_PreNorm_Net.
    Supports arbitrary state dimension n, suitable for high-dimensional
    sparse systems such as coupled Van der Pol oscillators.

    Each state dimension gets its own pair of AC channels:
      - g_i: state-update branch (learns dx_i/dt)
      - f_i: output-readout branch (not used for symbolic extraction)

    Architecture per channel:
      inputs_n -> MaskedLinearLayer(n+1, w1) -> MultiplicativeLayer -> velocity_i
    """
    def __init__(self, state_dim, w1, num_t_steps=10, dt=0.01, f_len=4):
        super(AC_Filter_PreNorm_Net_ND, self).__init__()
        self.state_dim = state_dim
        self.num_t_steps = num_t_steps
        self.dt = dt
        self.f_len = f_len

        num_in = state_dim + 1  # augmented with bias col

        # Normalization over all state dims; bias col excluded as always
        self.norm = FeatureNorm(num_features=state_dim)

        # One dynamics channel per state dimension
        # g_i learns the state update (dx_i/dt)
        # f_i learns the output readout (separate from dynamics)
        self.g_layers = nn.ModuleList([
            MaskedLinearLayer(size_in=num_in, size_out=w1)
            for _ in range(state_dim)
        ])
        self.f_layers = nn.ModuleList([
            MaskedLinearLayer(size_in=num_in, size_out=w1)
            for _ in range(state_dim)
        ])
        self.g_mult = nn.ModuleList([MultiplicativeLayer() for _ in range(state_dim)])
        self.f_mult = nn.ModuleList([MultiplicativeLayer() for _ in range(state_dim)])

        self.mha1 = nn.MultiheadAttention(num_in, num_heads=1, batch_first=True)

        # mask_v1 zeros out bias col in attn output; mask_v2 restores it to 1.0
        # shape [num_in] — generalizes to any state_dim
        mask_v1 = torch.ones(num_in)
        mask_v1[0] = 0.0
        mask_v2 = torch.zeros(num_in)
        mask_v2[0] = 1.0
        self.register_buffer('mask_v1', mask_v1)
        self.register_buffer('mask_v2', mask_v2)

        self.latent_state = None  # physical scale, for plotting
        self.latent_v = None      # physical scale, for plotting

    def _compute_velocity(self, state_n):
        """
        Returns:
          v_out:   list of [Batch, L, 1] output-readout velocities (f channels)
          v_state: list of [Batch, L, 1] state-update velocities (g channels)
        """
        v_out   = [self.f_mult[i](self.f_layers[i](state_n))
                   for i in range(self.state_dim)]
        v_state = [self.g_mult[i](self.g_layers[i](state_n))
                   for i in range(self.state_dim)]
        return v_out, v_state

    def _get_next_state(self, current_state_n, v_state):
        """
        Euler step in unit world.
        v_state: list of [Batch, L, 1], one per state dim.
        """
        # Stack state updates: [Batch, L, state_dim]
        delta_state = torch.cat(v_state, dim=2) * self.dt
        # Prepend zeros for bias col: [Batch, L, state_dim+1]
        z = torch.zeros(
            current_state_n.shape[0], current_state_n.shape[1], 1,
            device=current_state_n.device
        )
        delta = torch.cat((z, delta_state), dim=2)
        next_state_n = current_state_n + delta
        next_state_n[:, :, 0] = 1.0  # restore bias column
        return next_state_n

    def _to_physical(self, state_n, sigma):
        """Convert unit-world state [Batch, L, state_dim+1] to physical scale."""
        out = state_n.clone()
        out[..., 1:] = state_n[..., 1:] * (sigma + self.norm.eps)
        return out

    def _denorm_output(self, state_n, sigma):
        """Extract state cols and rescale to physical units: [Batch, L, state_dim]."""
        return state_n[..., 1:] * (sigma + self.norm.eps)

    def forward(self, t, inputs):
        # inputs: [Batch, L, state_dim+1] — cols [1.0, x_1, ..., x_n]
        L = inputs.size(1)
        attn_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=inputs.device), diagonal=1
        )

        # 1. Update sigma from raw state cols only
        if self.training:
            self.norm.update_stats(inputs[..., 1:])  # [Batch, L, state_dim]
        sigma = self.norm.running_sigma.detach()      # [state_dim]

        # 2. Normalize state cols before attention; bias col stays 1.0
        inputs_n = inputs.clone()
        inputs_n[..., 1:] = inputs[..., 1:] / (sigma + self.norm.eps)

        # 3. Causal self-attention on normalized inputs
        attn_out, _ = self.mha1(inputs_n, inputs_n, inputs_n,
                                 attn_mask=attn_mask, is_causal=True)

        # 4. Restore bias col; attn_out already unit-scale
        S1_n = (attn_out * self.mask_v1) + self.mask_v2
        # S1_n: [Batch, L, state_dim+1], unit world

        # 5. Rollout in unit world for f_len steps
        states_n = [S1_n]
        v_outs = []
        for step in range(self.f_len):
            v_out, v_state = self._compute_velocity(states_n[-1])
            v_outs.append(v_out)
            next_s = self._get_next_state(states_n[-1], v_state)
            states_n.append(next_s)
        # states_n[0] = S1_n (current), states_n[1..f_len] = predicted futures

        # 6. Store physical-scale quantities for plotting
        self.latent_state = self._to_physical(S1_n, sigma)
        self.latent_v = torch.cat(v_outs[0], dim=2) * (sigma + self.norm.eps)
        # latent_v: [Batch, L, state_dim], physical-scale velocity at step 0

        # 7. Denormalize predicted future states and concatenate
        # Each out_i: [Batch, L, state_dim]; final: [Batch, L, state_dim * f_len]
        outs = [self._denorm_output(states_n[i+1], sigma)
                for i in range(self.f_len)]
        return torch.cat(outs, dim=2)
    