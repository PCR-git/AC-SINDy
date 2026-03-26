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

################################################################################
################################################################################
    
# Dynamic Simulation Class
class DynamicSim():
  def __init__(self,X0,tf,t0,dt,params):
    self.Xi = X0.astype(float) # Initialize state vector to initial condition
    self.ti = t0 # Initial time
    self.tf = tf # Final time
    self.dt = dt # Time step size
    self.params = params # System parameters

  ############################################

  # The system's equations of motion.
  # This is a placeholder function which
  # can be overwritten for a given dynamical system.
  def f_Eq_of_Motion(self):
    print("Test DynamicSim class.")
    return 0

  ############################################

  # Simulate the dynamical system:
  def f_Simulate(self):

    self.Xi = self.Xi.astype(float)

    tf = self.tf
    dt = self.dt
    params = self.params

    Nt = int(tf/dt) # Number of time steps
    XN = np.zeros((len(self.Xi),Nt)) # Initialize state vector
    XP = np.zeros((len(self.Xi),Nt)) # Initialize vector of derivatives
    uN = np.zeros(Nt) # Initialize vector of control inputs
    t_vec = np.zeros((1,Nt)) # Initialize vector of times
    XN[:,0] = self.Xi # Initial conditions

    i = 0 # Initialize iteration counter
    # Iterate over time interval:
    for i in np.arange(Nt-1):
        # Update time derivative using the equations of motion
        Xp, uN[i] = self.f_Eq_of_Motion()
        Xp += params[i]
        self.Xi += Xp*dt # Update state at ith time step

        XN[:,i+1] = self.Xi # Update state
        XP[:,i] = Xp # Update time derivative
        t_vec[:,i] = self.ti # Update time vector
        i += 1 # Increment counter
        self.ti += dt # Increment timer

        # If system diverges, break out of simulation:
        if np.max(np.abs(XN)) > 1e10:
          break

    XP[:,i], uN[i] = self.f_Eq_of_Motion() # Update final derivative and control
    t_vec[:,i] = self.ti # Update time vector

    # Ouput: Nominal state & derivative (only up to (i+1)th entry, in case of divergence)
    return XN[:,0:i+1], XP[:,0:i+1], uN[0:i+1], t_vec[:,0:i+1]

  ############################################

  # Run num_sim number of simulations:
  def f_N_Simulations(self,X0_tot,num_sims):

    tf = self.tf
    dt = self.dt
    params = self.params
    dim = np.shape(self.Xi)[0]
    Nt = int(tf/dt) # Number of time steps

    # Initialize arrays to hold data from all simulations:
    XN_tot = np.zeros((dim,Nt*num_sims)) # Initialize state vector
    XP_tot = np.zeros((dim,Nt*num_sims)) # Initialize vector of derivatives
    uN_tot = np.zeros(Nt*num_sims) # Initialize vector of control inputs
    t_vec_tot = np.zeros((1,Nt*num_sims)) # Initialize vector of times

    ct = 0 # Total number of time steps elapsed
    for i in np.arange(num_sims):
      X0 = X0_tot[i] # Get initial condition
      self.Xi = X0.astype(float) # Update initial condition

      XN, XP, uN, t_vec = self.f_Simulate() # Run a simulation

      lt = np.shape(t_vec)[1] # Number of time steps in simulation
      ct2 = ct + lt # Compute total number of time steps so far

      # Add simulation results to arrays
      XN_tot[:,ct:ct2] = XN
      XP_tot[:,ct:ct2] = XP
      uN_tot[ct:ct2] = uN
      t_vec_tot[:,ct:ct2] = t_vec
      ct = ct2 # Update total number of time steps so far

    return XN_tot, XP_tot, uN_tot, t_vec_tot

############################################

class Nonlinear_Damped_Osc_2D(DynamicSim):
    def __init__(self,X0,tf,t0,dt,params):
        super(Nonlinear_Damped_Osc_2D, self).__init__(X0,tf,t0,dt,params)

    def f_Eq_of_Motion(self):
        Xi = self.Xi
        out = np.array([-0.1*Xi[0] + Xi[1], -2*Xi[0] - 0.1*Xi[1] - 0.5*Xi[0]*Xi[1] - 0.025*Xi[1]**2])
        return out, 0


