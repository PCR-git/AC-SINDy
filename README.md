## Compositional Neural SINDy: Scalable Sparse Identification of Nonlinear Dynamics

A compositional neural extension of the Sparse Identification of Nonlinear Dynamics (SINDy) framework for learning sparse, interpretable dynamical systems.

The original SINDy algorithm (Brunton et al., 2016) identifies governing equations by selecting a sparse combination of candidate functions. This project replaces the explicit function library with a structured neural architecture, enabling more expressive and scalable representations.

---

## Repository Structure

```
neural-sindy/
├── model/
│   ├── layers.py          # masked linear, Hadamard layer, etc.
│   ├── models.py          # SINDy models
│   ├── pruning.py         # pruning logic
│   ├── train.py           # training loop
│   ├── dynamics_sim.py    # system simulation (Lorenz, etc.)
│   ├── utils.py
│   └── plot.py
│
├── neural_sindy.ipynb     # Experiments on dynamical systems
│
├── neural_sindy.pdf       # Technical report (method + results)
│
├── README.md
```

---

## Method Overview

- The model consists of compositional neural layers (e.g., linear and multiplicative/Hadamard layers) that generate candidate functions.
- The network is trained to fit system dynamics from data.
- Parameters are iteratively pruned based on their marginal impact on model performance.
- This process is repeated until performance degrades, yielding a sparse and interpretable model.

---

## Key Ideas

- **Compositional function representation**
  - Replaces explicit SINDy libraries with a multi-layer architecture that builds nonlinear functions via composition.
  - Enables representation of exponentially many candidate terms with a polynomial number of parameters.

- **Pruning-based model selection**
  - Iteratively removes parameters based on their impact on model performance.
  - Avoids biases introduced by magnitude-based (L1) sparsification.

- **Multi-step training (Neural ODE formulation)**
  - Uses trajectory-level supervision rather than single-step regression.
  - Helps disambiguate competing models and improves robustness.

- **Scalability**
  - Standard SINDy requires \(O(n^p)\) candidate terms for polynomial order \(p\).
  - Neural SINDy achieves comparable expressivity with \(O(n^2 p)\) parameters under sparsity assumptions.

---

## Experiments

The notebook `neural_sindy.ipynb` contains experiments on:

- 2D dynamical systems
- The Lorenz system (chaotic 3D dynamics)
- Systems with nonlinear and compositional structure (e.g., sinusoidal forcing)

Results demonstrate that the method can:
- recover accurate governing equations
- identify sparse and interpretable models
- maintain strong predictive performance under pruning

---

## Scaling Insight

Standard SINDy requires explicit enumeration of candidate functions, which grows combinatorially as \(O(n^p)\).  
In contrast, the compositional architecture represents these functions implicitly using shared parameters, reducing complexity to \(O(n^2 p)\) under sparsity assumptions.

See the technical report (`neural_sindy.pdf`) for full derivations and analysis.

---

## References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).  
  *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*  
  Proceedings of the National Academy of Sciences.  
  https://www.pnas.org/doi/10.1073/pnas.1517384113
