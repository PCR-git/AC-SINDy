import torch
import numpy as np

from sympy import symbols, expand, nsimplify, Float

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

################################################################################
################################################################################ 

# def f_extract_symbolic(model, input_names=['1', 'x', 'y']):
#     """
#     Extracts and expands the symbolic ODE equations from a 2D Arithmetic Circuit.
#     Uses sympy to expand the factored compositional representation into
#     individual polynomial terms with coefficients rounded to 4 decimal places.
#     """
#     model.eval()
#     scaling_factor = getattr(model, 'std_dev', 1.0)

#     # Create sympy symbols — '1' becomes the integer 1
#     sym_inputs = []
#     for name in input_names:
#         if name == '1':
#             sym_inputs.append(1)
#         else:
#             sym_inputs.append(symbols(name))

#     def get_layer_expressions(layer):
#         weights = (layer.weight / scaling_factor).detach().cpu().numpy()
#         mask = layer.mask.detach().cpu().numpy()
#         active_weights = weights * mask

#         exprs = []
#         for row in active_weights:
#             expr = sum(round(float(w), 4) * s 
#                        for w, s in zip(row, sym_inputs) if abs(w) > 1e-4)
#             exprs.append(expr if expr != 0 else 0)
#         return exprs

#     with torch.no_grad():
#         h1_exprs = get_layer_expressions(model.h1)
#         h2_exprs = get_layer_expressions(model.h2)

#         m1_expr = expand(1)
#         for e in h1_exprs:
#             m1_expr = expand(m1_expr * e)

#         m2_expr = expand(1)
#         for e in h2_exprs:
#             m2_expr = expand(m2_expr * e)

#         # Round all coefficients in the expanded expression to 4 decimal places
#         m1_expr = m1_expr.xreplace({n: round(n, 4) for n in m1_expr.atoms(Float)})
#         m2_expr = m2_expr.xreplace({n: round(n, 4) for n in m2_expr.atoms(Float)})

#         print("--- Recovered Governing Equations ---")
#         print(f"dx/dt = {m1_expr}")
#         print(f"dy/dt = {m2_expr}")
#         print("---------------------------------------")

## Symbolic Extraction Functions

################################################################################
# Shared helpers
################################################################################

def _get_sym_inputs(input_names):
    """Convert input name list to sympy symbols, treating '1' as the integer 1."""
    sym_inputs = []
    for name in input_names:
        if name == '1':
            sym_inputs.append(1)
        else:
            sym_inputs.append(symbols(name))
    return sym_inputs

def _layer_to_expr(layer, sym_inputs, scaling_factor=1.0):
    """
    Convert a MaskedLinearLayer to a list of sympy expressions,
    one per output feature (row of the weight matrix).
    """
    weights = (layer.weight / scaling_factor).detach().cpu().numpy()
    mask = layer.mask.detach().cpu().numpy()
    active_weights = weights * mask

    exprs = []
    for row in active_weights:
        expr = sum(round(float(w), 4) * s
                   for w, s in zip(row, sym_inputs) if abs(w) > 1e-4)
        exprs.append(expr if expr != 0 else 0)
    return exprs

def _expand_and_round(exprs):
    """Multiply a list of expressions together, expand, and round all coefficients."""
    result = expand(1)
    for e in exprs:
        result = expand(result * e)
    return result.xreplace({n: round(n, 4) for n in result.atoms(Float)})

def _print_equations(equations, state_names):
    print("--- Recovered Governing Equations ---")
    for name, expr in zip(state_names, equations):
        print(f"d{name}/dt = {expr}")
    print("---------------------------------------")

################################################################################
# AC_Net_2D
################################################################################

def f_extract_symbolic_2D(model, input_names=['1', 'x', 'y']):
    """
    Extract symbolic equations from AC_Net_2D.
    Architecture: inputs -> (h1, h2) -> (m1, m2) -> (dx, dy)
    Output col 0 is zeroed; cols 1 and 2 are dx and dy.
    """
    model.eval()
    scaling_factor = getattr(model, 'std_dev', 1.0)
    sym_inputs = _get_sym_inputs(input_names)

    with torch.no_grad():
        h1_exprs = _layer_to_expr(model.h1, sym_inputs, scaling_factor)
        h2_exprs = _layer_to_expr(model.h2, sym_inputs, scaling_factor)

        dx = _expand_and_round(h1_exprs)
        dy = _expand_and_round(h2_exprs)

    _print_equations([dx, dy], ['x', 'y'])

################################################################################
# AC_Net_3D
################################################################################

def f_extract_symbolic_3D(model, input_names=['1', 'x', 'y', 'z']):
    """
    Extract symbolic equations from AC_Net_3D.
    Architecture: inputs -> (h1, h2, h3) multiplicative branches
                         + (h4, h5, h6) linear residual branches
                  -> (dx, dy, dz)
    Each output is the product branch plus the residual: m(hi) + hi_res.
    """
    model.eval()
    scaling_factor = getattr(model, 'std_dev', 1.0)
    sym_inputs = _get_sym_inputs(input_names)

    with torch.no_grad():
        # Multiplicative branches
        h1_exprs = _layer_to_expr(model.h1, sym_inputs, scaling_factor)
        h2_exprs = _layer_to_expr(model.h2, sym_inputs, scaling_factor)
        h3_exprs = _layer_to_expr(model.h3, sym_inputs, scaling_factor)

        dx_prod = _expand_and_round(h1_exprs)
        dy_prod = _expand_and_round(h2_exprs)
        dz_prod = _expand_and_round(h3_exprs)

        # Linear residual branches (h4, h5, h6 each have size_out=1)
        h4_exprs = _layer_to_expr(model.h4, sym_inputs, scaling_factor)
        h5_exprs = _layer_to_expr(model.h5, sym_inputs, scaling_factor)
        h6_exprs = _layer_to_expr(model.h6, sym_inputs, scaling_factor)

        # Residuals are single-output layers so take the first (only) expression
        dx_res = h4_exprs[0] if h4_exprs else 0
        dy_res = h5_exprs[0] if h5_exprs else 0
        dz_res = h6_exprs[0] if h6_exprs else 0

        dx = expand(dx_prod + dx_res).xreplace(
            {n: round(n, 4) for n in expand(dx_prod + dx_res).atoms(Float)})
        dy = expand(dy_prod + dy_res).xreplace(
            {n: round(n, 4) for n in expand(dy_prod + dy_res).atoms(Float)})
        dz = expand(dz_prod + dz_res).xreplace(
            {n: round(n, 4) for n in expand(dz_prod + dz_res).atoms(Float)})

    _print_equations([dx, dy, dz], ['x', 'y', 'z'])

################################################################################
# AC_Filter
################################################################################

def f_extract_symbolic_filter(model, input_names=['1', 'x', 'y']):
    """
    Extract symbolic equations from AC_Filter_Net.
    The dynamics are encoded in the g1/g2 layers (state-update branches).
    f1/f2 are output-readout branches and do not correspond to the governing
    equations, so they are ignored.
    No sigma rescaling needed — AC_Filter_Net operates in physical scale throughout.
    """
    model.eval()
    sym_inputs = _get_sym_inputs(input_names)

    with torch.no_grad():
        g1_exprs = _layer_to_expr(model.g1, sym_inputs)
        g2_exprs = _layer_to_expr(model.g2, sym_inputs)

        dx = _expand_and_round(g1_exprs)
        dy = _expand_and_round(g2_exprs)

    _print_equations([dx, dy], ['x', 'y'])
    
################################################################################
# AC_Filter_PreNorm_Net
################################################################################

def f_extract_symbolic_filter_prenorm(model, input_names=['1', 'x', 'y']):
    """
    Extract symbolic equations from AC_Filter_PreNorm_Net.
    The g1/g2 layers learn dynamics in unit world, so their coefficients
    are rescaled by sigma to recover physical-scale governing equations.
    """
    model.eval()
    sym_inputs = _get_sym_inputs(input_names)

    sigma = model.norm.running_sigma.detach().cpu().numpy()  # [2]

    with torch.no_grad():
        g1_exprs = _layer_to_expr(model.g1, sym_inputs)
        g2_exprs = _layer_to_expr(model.g2, sym_inputs)

        dx_unit = _expand_and_round(g1_exprs)
        dy_unit = _expand_and_round(g2_exprs)

        dx = expand(dx_unit * round(float(sigma[0]), 4)).xreplace(
            {n: round(n, 4) for n in
             expand(dx_unit * round(float(sigma[0]), 4)).atoms(Float)})
        dy = expand(dy_unit * round(float(sigma[1]), 4)).xreplace(
            {n: round(n, 4) for n in
             expand(dy_unit * round(float(sigma[1]), 4)).atoms(Float)})

    _print_equations([dx, dy], ['x', 'y'])
    