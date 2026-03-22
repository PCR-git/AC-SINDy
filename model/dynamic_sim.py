import torch
import torch.nn as nn
import numpy as np

################################################################################
################################################################################

class Lorenz(nn.Module):
    """
    Lorenz System
    Given current position, outputs current velocity.
    """
    def __init__(self, params):
        super().__init__()
        # Convert numpy array to tensor and register as a buffer.
        self.register_buffer('p', torch.tensor(params, dtype=torch.float32))

    def forward(self, t, X):
        # Unpack parameters from the on-device buffer
        sigma, beta, rho, A, omega = self.p[0], self.p[1], self.p[2], self.p[3], self.p[4]

        # X is typically [1, 3] or [Batch, 3]
        # Using x, y, z notation for clarity
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        # Control term
        u = A * torch.sin(omega * t) 

        # Define equations of motion
        dx = sigma * (y - x)
        dy = x * (rho - z) - y + u  # Adding control 'u' here as an example
        dz = x * y - beta * z

        return torch.stack([dx, dy, dz], dim=1)

################################################################################
################################################################################

def f_Random_Matrix(n,rng,omega=1):
    u = rng.choice(np.arange(n**2),size=n)
    xi = np.floor(u/n).astype(int)
    xj = np.mod(u,n)

    G = np.zeros((n,n))
    for i in np.arange(n):
        G[xi[i],xj[i]] = -1*rng.uniform(low=0.0, high=1.0)

    for i in np.arange(n):
        G[i,i] = omega

    for i in np.arange(n-1):
        G[i,i+1] = -1*rng.uniform(low=0.0, high=1.0)

    G[n-1,0] = -1*rng.uniform(low=0.0, high=1.0)

    return G

################################################################################
################################################################################

class Van_der_Pol_osc(nn.Module):
    """
    Coupled Van der Pol oscillators.
    Calculates derivatives for a system of coupled non-linear oscillators.
    """
    def __init__(self, params, K):
        super().__init__()
        self.register_buffer('K', torch.tensor(K, dtype=torch.float32))
        self.params = params

    def forward(self, t, X):
        mu = self.params[0]

        sz = X.shape[0]
        dim = int(sz / 2)
        x = X[0:dim]
        y = X[dim:]

        xp = y
        yp = mu * (1 - x**2) * y - torch.matmul(self.K, x)

        return torch.cat((xp, yp))


