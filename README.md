## AC-SINDy: Compositional Sparse Identification of Nonlinear Dynamics

**TL;DR:** Learn sparse dynamical systems without enumerating candidate functions.

A compositional extension of the Sparse Identification of Nonlinear Dynamics (SINDy) framework that replaces explicit feature libraries with a structured, factorized representation.

Standard SINDy identifies governing equations by selecting sparse combinations of candidate functions from a predefined library. However, this library grows combinatorially with system dimension and nonlinearity.

AC-SINDy avoids explicit enumeration by constructing functions implicitly through compositions of simple primitives (e.g., linear transformations and multiplicative interactions), yielding a compact and scalable representation.

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


---

## Method Overview

- The model is a **compositional architecture** consisting of alternating linear and multiplicative layers.
- These layers define a structured computational graph that builds nonlinear features through composition.
- The resulting representation can be viewed as a **factorized parameterization of polynomial interactions**.
- The model is trained on trajectory data (Neural ODE formulation).
- Sparsity is induced through **iterative pruning with gradient-based importance estimates**.

---

## Key Ideas

### **Compositional function representation**
- Replaces explicit SINDy libraries with an implicit, learned function class.
- Nonlinear terms are constructed via compositions of simple primitives.
- Enables representation of complex interactions without enumerating all candidate terms.

### **Factorized hypothesis class**
- The model represents functions as compositions of low-order operations.
- Equivalent to a **factorized (low-rank) representation of polynomial features**.
- Trades completeness of the function library for parameter efficiency and structure.

### **Structure learning via pruning**
- Parameters correspond to edges in the computational graph.
- Iterative pruning removes low-importance connections using gradient-based criteria.
- Produces sparse and interpretable models.

### **Feature Normalization**
- Each feature is normalized independently using running statistics.
- Ensures coefficients reflect true importance rather than feature scale.
- Provides a scale-invariant parameterization while preserving interpretability.

### **Scalability through factorization**
- Standard SINDy requires \(O(n^p)\) candidate terms for polynomial order \(p\).
- AC-SINDy replaces this with a compositional representation.
- Under sparsity assumptions, parameter count scales as \(O(n^2 p)\).

---

## Experiments

The notebook `ac_sindy.ipynb` includes experiments on:

- 2D nonlinear dynamical systems
- The Lorenz system (chaotic 3D dynamics)
- Systems with nonlinear and compositional structure (e.g., sinusoidal forcing)

Results demonstrate that the method can:
- recover governing equations with correct functional structure
- identify sparse and interpretable representations
- maintain strong predictive performance under pruning

---

## Conceptual Insight

Standard SINDy relies on **explicit basis expansion**:
- enumerates candidate functions
- scales combinatorially with complexity

AC-SINDy replaces this with **compositional factorization**:
- constructs functions through structured composition
- avoids explicit enumeration
- provides a compact, structured alternative for system identification

---

## References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).  
  *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*  
  Proceedings of the National Academy of Sciences.  
  https://www.pnas.org/doi/10.1073/pnas.1517384113
