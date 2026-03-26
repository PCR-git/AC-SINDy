import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

################################################################################
################################################################################

# class MultiplicativeLayer(nn.Module):
#     """
#     Computes the product of all features across the final dimension.
    
#     This layer is typically used in SINDy-based architectures to generate 
#     higher-order polynomial terms from input features.
#     """
#     def __init__(self):
#         super(MultiplicativeLayer, self).__init__()

#     def forward(self, inputs):
#         return torch.prod(inputs, dim=2, keepdim=True)

class MultiplicativeLayer(nn.Module):
    def __init__(self):
        super(MultiplicativeLayer, self).__init__()

    def forward(self, inputs):
        # Determine which dimension to multiply based on input rank
        # If 3D [Batch, Seq, Dim], use dim=2
        # If 2D [Batch, Dim], use dim=1
        target_dim = inputs.ndim - 1
        return torch.prod(inputs, dim=target_dim, keepdim=True)
    
################################################################################
################################################################################

class HadamardLayer(nn.Module):
    """ 
    Performs a masked element-wise (Hadamard) multiplication between inputs 
    and a learnable weight matrix.
    
    Args:
        size_in (int): Number of input features.
        size_out (int): Number of output features.
        use_bias (bool): If True, appends a column of ones to the input for bias terms.
    """
    def __init__(self, size_in, size_out, use_bias=False):
        super().__init__()

        self.use_bias = use_bias
        if self.use_bias == True:
          # self.bias = nn.Parameter(torch.Tensor((1,1)))
          # self.bias.data = nn.parameter.Parameter(torch.tensor([[0.]]))
          size_in += 1
          # size_out += 1

        self.register_buffer('mask', torch.ones((size_out, size_in), dtype=torch.bool))

        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.
        # init_array = torch.ones((size_out, size_in))
        # self.weight.data = nn.parameter.Parameter(init_array)
        torch.nn.init.kaiming_uniform_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.normal_(self.weight, mean=1, std=1)

    def forward(self, x):
        if self.use_bias == True:
            # FIX: Match the device of x
            bias_col = torch.ones((x.size()[0], x.size()[1], 1), device=x.device)
            x = torch.cat((x, bias_col), dim=2)

        hx = x * self.weight * self.mask
        return hx

    # Compute mask
    def compute_mask(self,idx):
        "Set mask[idx] to False"
        self.mask[idx] = False # Set mask term to zero
        with torch.no_grad(): # (zeroing out weight isn't differentiable)
            self.weight[idx] = 0 # Set weight term to zero
    
################################################################################
################################################################################

# Masked Linear Layer
class MaskedLinearLayer(torch.nn.Linear):
    """
    Standard linear layer augmented with a persistent pruning mask.
    
    Inherits from torch.nn.Linear but applies a binary mask to weights 
    during the forward pass to facilitate network sparsification.
    """
    def __init__(self, size_in: int, size_out: int, use_bias=False, keep_layer_input=False):
        super().__init__(size_in, size_out, use_bias)
        self.register_buffer('mask', torch.ones((size_out, size_in), dtype=torch.float32))
        self.keep_layer_input = keep_layer_input
        self.layer_input = None

    def forward(self, input):
        x = input.float()
        if self.keep_layer_input:
            self.layer_input = x.detach()

        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

    def compute_mask(self, idx):
        self.mask[idx] = 0.0
        with torch.no_grad():
            self.weight[idx] = 0.0

################################################################################
################################################################################            

# # Masked Linear Layer 2
# class MaskedLinearLayer2(torch.nn.Linear):
#     """
#     Alternative Masked Linear Layer supporting specialized activation constraints.
    
#     Provides options for 'sigmoid' or 'abs' constrained weights, often 
#     used for maintaining positivity or stability in dynamical system discovery.
#     """
#     def __init__(self, size_in: int, size_out: int, use_bias=False, keep_layer_input=False, act=False):
#         super().__init__(size_in, size_out, use_bias)
#         self.register_buffer('mask', torch.ones((size_out, size_in), dtype=torch.float32))
#         self.keep_layer_input = keep_layer_input
#         self.act = act

#         nn.init.normal_(self.weight, mean=1, std=1)

#     def forward(self, input):
#         x = input.float()
#         if self.keep_layer_input:
#             self.layer_input = x.detach()

#         if self.act == 'sigmoid':
#             w = torch.sigmoid(self.weight)
#         elif self.act == 'abs':
#             w = torch.abs(self.weight)
#         else:
#             w = self.weight
            
#         # Apply mask
#         w = w * self.mask

#         out = torch.matmul(x, w.t())

#         if self.bias is not None:
#             out += self.bias

#         return out.unsqueeze(1) if out.dim() == 2 else out 
    
#     def compute_mask(self, idx):
#         """
#         Zeroes out the mask and the weight at the specified index.
#         This effectively prunes the connection from the network.
#         """
#         # Set mask term to zero (1.0 -> 0.0)
#         self.mask[idx] = 0.0 
        
#         with torch.no_grad(): # Ensure this change doesn't interfere with gradients
#             # Set the weight itself to zero so it doesn't contribute to the sum
#             self.weight[idx] = 0.0

class MaskedLinearLayer2(torch.nn.Linear):
    """
    Masked linear layer with optional weight constraints.
    Supports 'sigmoid' or 'abs' activation on weights for use in
    architectures requiring positive or bounded weight values,
    such as rational dynamical system discovery (HLMSD_Net_2D).
    """
    def __init__(self, size_in: int, size_out: int, use_bias=False,
                 keep_layer_input=False, act=False):
        super().__init__(size_in, size_out, use_bias)
        self.register_buffer('mask', torch.ones((size_out, size_in), dtype=torch.float32))
        self.keep_layer_input = keep_layer_input
        self.act = act
        nn.init.kaiming_uniform_(self.weight, a=1, mode='fan_in',
                                  nonlinearity='leaky_relu')

    def forward(self, input):
        x = input.float()
        if self.keep_layer_input:
            self.layer_input = x.detach()

        if self.act == 'sigmoid':
            w = torch.sigmoid(self.weight)
        elif self.act == 'abs':
            w = torch.abs(self.weight)
        else:
            w = self.weight

        w = w * self.mask
        out = torch.matmul(x, w.t())
        if self.bias is not None:
            out += self.bias
        return out

    def compute_mask(self, idx):
        self.mask[idx] = 0.0
        with torch.no_grad():
            self.weight[idx] = 0.0
            
################################################################################
################################################################################              
  
class MaskedHadamardLinearLayer(nn.Module):
    """ 
    A composite layer sequencing a Hadamard Layer followed by a Masked Linear Layer.

    This structure is useful for learning sparse, nonlinear coefficient 
    dependencies in SINDy models.
    """
    def __init__(self,size_in,size_out,use_bias=False):
        super().__init__()
        self.size_in = size_in
        self.h = HadamardLayer(size_in, size_out,use_bias=use_bias)
        self.l = MaskedLinearLayer2(size_in, size_out,use_bias=use_bias)

    def forward(self, x):
        # x = x.view(-1, self.size_in)
        hx = self.h(x)
        # hx = f_layer_norm(hx)
        lx = self.l(hx)
        return lx            

################################################################################
################################################################################  

class Power(nn.Module):
    """
    Raises input elements to a learnable exponent 'b'.
    
    Used to discover non-integer power laws in dynamical systems.
    """
    def __init__(self):
        super(Power, self).__init__()
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, inputs):
        return torch.pow(inputs, self.b)

################################################################################
################################################################################  


