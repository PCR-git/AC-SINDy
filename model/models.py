import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import MaskedLinearLayer, MaskedLinearLayer2, MultiplicativeLayer, MaskedHadamardLinearLayer
from .model_utils import f_get_linear_layers

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

    def initialize_rand_normal(self, mean=0, std=0.1):
        layers = f_get_linear_layers(self)
        for layer in layers:
            nn.init.normal_(layer.weight, mean=mean, std=std)
            nn.init.constant_(layer.bias, val=0)

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

    def initialize_rand_normal(self, mean=0, std=0.1):
        layers = f_get_linear_layers(self)
        for layer in layers:
            nn.init.normal_(layer.weight, mean=mean, std=std)
            nn.init.constant_(layer.bias, val=0)

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

    def initialize_rand_normal(self, mean=0, std=0.1):
        layers = f_get_linear_layers(self)
        for layer in layers:
            nn.init.normal_(layer.weight, mean=mean, std=std)
            nn.init.constant_(layer.bias, val=0)
            
