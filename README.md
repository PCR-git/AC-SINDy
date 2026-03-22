## AC-SINDy: Scalable Sparse Identification of Nonlinear Dynamics via Arithmetic Circuits

A compositional extension of the Sparse Identification of Nonlinear Dynamics (SINDy) framework that replaces explicit feature libraries with structured arithmetic circuits for learning sparse, interpretable dynamical systems.

The original SINDy algorithm (Brunton et al., 2016) identifies governing equations by selecting a sparse combination of candidate functions from a predefined library. In contrast, this project constructs candidate functions implicitly through compositions of linear transformations and multiplicative interactions, enabling a more scalable and structured representation.

---

## Repository Structure

```
ac-sindy/
├── model/
│   ├── layers.py          # masked linear, Hadamard layer, etc.
│   ├── models.py          # SINDy models
│   ├── pruning.py         # pruning logic
│   ├── train.py           # training loop
│   ├── dynamics_sim.py    # system simulation (Lorenz, etc.)
│   ├── utils.py
│   └── plot.py
│
├── ac_sindy.ipynb     # Experiments on dynamical systems
│
├── ac_sindy.pdf       # Technical report (method + results)
│
├── README.md
```

---

## Method Overview

- The model is a **compositional architecture** consisting of alternating linear and multiplicative layers.
- These layers form a **sparse arithmetic circuit**, constructing nonlinear features via compositions of sums and products.
- The network is trained to fit system dynamics from data.
- Parameters are iteratively pruned based on their marginal impact on performance.
- The result is a **sparse, structured representation** of the governing equations.

---

## Key Ideas

### **Arithmetic circuit representation**
- Replaces explicit SINDy libraries with a compositional computational graph.
- Nonlinear features are constructed via products of learned linear functions.
- Equivalent to learning a **factorized representation of polynomial interactions**, rather than enumerating all terms.

### **Structure learning via pruning**
- Each parameter corresponds to an edge in the computational graph.
- Iterative pruning removes edges with minimal impact on performance.
- Produces a sparse and interpretable model structure.

### **Multi-step training (Neural ODE formulation)**
- Uses trajectory-level supervision instead of single-step regression.
- Encourages consistency over time and improves robustness.

### **Scalability through factorization**
- Standard SINDy requires \(O(n^p)\) candidate terms for polynomial order \(p\).
- AC-SINDy replaces basis expansion with compositional factorization.
- Under sparsity assumptions, parameter count scales as \(O(n^2 p)\).

---

## Experiments

The notebook `ac_sindy.ipynb` contains experiments on:

- 2D dynamical systems
- The Lorenz system (chaotic 3D dynamics)
- Systems with nonlinear and compositional structure (e.g., sinusoidal forcing)

Results demonstrate that the method can:
- recover accurate governing equations
- identify sparse and interpretable models
- maintain strong predictive performance under pruning

---

## Conceptual Insight

Standard SINDy relies on **basis expansion**:
- explicitly enumerates candidate functions
- scales combinatorially with system complexity

AC-SINDy replaces this with **compositional factorization**:
- constructs functions through structured compositions
- avoids explicit enumeration
- trades completeness for scalability and structure

---

## References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).  
  *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*  
  Proceedings of the National Academy of Sciences.  
  https://www.pnas.org/doi/10.1073/pnas.1517384113
