## Neural SINDy

This repo contains a neural implementation of the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm.

The original SINDy (Sparse Identification of Nonlinear Dynamical systems) algorithm, introduced by Steven Brunton, J. Nathan Kutz,
and collaborators in 2016, is a method for discovering explicit governing equations of dynamical systems directly from data.
It combines the expressive power of a large dictionary of candidate nonlinear functions with sparse regression to recover interpretable,
low-dimensional models of the system's underlying dynamics. The original SINDy paper is available at: https://www.pnas.org/doi/10.1073/pnas.1517384113

I independently developed the idea of implementing SINDy as a neural network and wrote this code in early 2023, but later discovered a similar
work had already appeared in the literature. As a result, I did not publish this work.

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
├── neural_sindy.ipynb   # Experiments on dynamical systems
│
├── README.md
```

---

The algorithm work as follows:
  - We have several neural network layers, implementing different function primitives, including a masked linear layer, with the mask allowing us to implement a form of pruning, and a multiplicative (Hadamard) layer that allows us to combine linear layers into higher-order polynomials.
  - We train the network until convergence, and then prune the least important weight, i.e the weight that, when removed, has the least impact on worsening the performance of the trained model.
  - We repeat this process, training the pruned model until convergence, pruning again, etc, continuing this process until the loss stops decreasing.

This repo contains a single ipynb file, which includes an implementation of the algorithm, functions for tracking learning & pruning, and
experiments with a simple 2D ODE, the Lorenz system (a chaotic 3D ODE), and a system with sinusoidal forcing.

---

Among our improvements to the SINDy algorithm are:
- A neural-network architecture (can automatically find constants and learn more complicated dynamics).
  - Parameter sharing allows us to represent an exponential number of candidate functions with a polynomial number of parameters.
  - Makes the optimization easier by giving more paths down to equilibrium.
  - Makes high-dimensional systems and systems with highly-compositional dynamics tractable.
  - Makes use of neural network libraries and GPU computation.
  - SINDy-PI requires guessing and checking LHS terms, which hugely increases run-time, whereas we can represent fractional dynamics directly.
  - Adam appears to work well in practice, which is faster than second-order optimization.
- Use of Neural ODEs.
  - Predicts multiple time steps ahead and backpropagates through time, which further constrains the SINDy optimization.
  - Does so in a scalable manner by making use of the adjoint sensitivity method, and allows us to make use of powerful ODE solvers.
- Improved optimization.
  - We replace the L1 pruning criterion used by SINDy. Instead, we remove the weight which, when removed, has the least impact on worsening the performance of the trained model.
  - Using validation data to prune improves robustness.
  - Dummy variables allow us to apply L1 regularization.
  - Using batch-norm improves optimization by alleviating scaling/dimensionality issues, while still allowing us to recover weights.
  - Removing one variable at a time removes an unnecessary hyperparameter and provides a natural way of combining search with gradient-based optimization.
- Neural network interpretability through libraries of known functions and pruning.

The major advantage of our approach is that we can represent exponentially many terms with a quadratic number of parameters. In particular, suppose that the inputs $x \in \mathbb{R}^n$.
The number of polynomial terms of order $p$ is $\binom{n + p - 1}{p}.$ Hence, the number of polynomial terms of order $p$ or less is $\sum_{k=0}^p \binom{n + k - 1}{k} = \frac{1}{n} (p + 1) \binom{n + p}{p + 1} \sim O(p n^p)$ (for large $n$), which we can represent with as few as $p(n + 1)$ parameters using a multi-layer architecture. We can represent $m$ such terms with at worst $mp(n+1)$ parameters. (At worst, because if any of these terms involve one of the other terms, we get it for free.) A reasonable sparsity assumption is that the number of active terms is linear in the number of dimensions, i.e., $m = O(n)$, in which case we need $O(n^2 p)$ parameters. Hence, we beat regular SINDy for $p > 2$.

