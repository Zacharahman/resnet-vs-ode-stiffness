import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """ODE function f(t,x) that defines the vector field."""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, t, x):
        """Compute dx/dt = f(t,x)."""
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, solver='dopri5'):
        super().__init__()
        self.state_dim = state_dim
        self.odefunc = ODEFunc(state_dim, hidden_dim)
        self.solver = solver
        
    def forward(self, x0, t_span=[0., 1.]):
        """Solve ODE from t_span[0] to t_span[1] with adaptive stepping."""
        t = torch.tensor(t_span)
        solution = odeint(self.odefunc, x0, t, method=self.solver)
        return solution[-1]  # Return final state
    
    def trajectory(self, x0, t_eval):
        """Generate trajectory at specified time points."""
        t = torch.tensor(t_eval)
        solution = odeint(self.odefunc, x0, t, method=self.solver)
        return solution