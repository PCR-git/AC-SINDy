import torch
import torch.nn as nn

################################################################################
################################################################################

# class FeatureNorm(nn.Module):
#     def __init__(self, num_features, momentum=0.1, eps=1e-5):
#         super(FeatureNorm, self).__init__()
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
        
#         # We use buffers so these aren't updated by the optimizer
#         self.register_buffer('running_sigma', torch.ones(num_features))

#     def forward(self, x):
#         """
#         x: Tensor of shape (batch, num_features) or (time, batch, num_features)
#         """
#         if self.training:
#             # Calculate standard deviation of current batch
#             # Note: We flatten if input is (T, B, N) to get global stats
#             current_sigma = torch.sqrt(x.var(dim=tuple(range(x.ndim - 1))) + self.eps)
            
#             # Update running estimate (EMA)
#             with torch.no_grad():
#                 self.running_sigma = (1 - self.momentum) * self.running_sigma + \
#                                      self.momentum * current_sigma

#         # CRITICAL: Use detach() so gradients don't flow through the scaling factor
#         # This forces the optimizer to adjust the WEIGHTS, not the NORM.
#         scale = self.running_sigma.detach()
        
#         return x / (scale + self.eps)

#     def extra_repr(self):
#         return f'num_features={self.num_features}, momentum={self.momentum}'
    

class FeatureNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(FeatureNorm, self).__init__()
        self.num_features = num_features  # 2: x and y only
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_sigma', torch.ones(num_features))

    def update_stats(self, x):
        """Update running std from a [Batch, L, num_features] tensor."""
        with torch.no_grad():
            dims = tuple(range(x.ndim - 1))
            current_sigma = torch.sqrt(x.var(dim=dims) + self.eps)
            self.running_sigma = (1 - self.momentum) * self.running_sigma + \
                                 self.momentum * current_sigma

    def forward(self, x):
        scale = self.running_sigma.detach()
        return x / (scale + self.eps)
    
    