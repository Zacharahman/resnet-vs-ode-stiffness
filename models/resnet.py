import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, num_layers=4, dt=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        
        # Approximates the vector field f(t,x)
        self.f = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.num_layers = num_layers
    
    def forward(self, x0):
        """Forward pass using Euler discretization with fixed step size."""
        x = x0
        
        # Apply num_layers Euler steps
        for _ in range(self.num_layers):
            # Euler step: x_{n+1} = x_n + dt * f(x_n)
            dx = self.f(x)
            x = x + self.dt * dx
            
        return x
    
    def trajectory(self, x0, num_steps):
        """Generate trajectory by applying forward pass multiple times."""
        trajectory = [x0]
        x = x0
        
        for _ in range(num_steps):
            x = self.forward(x)
            trajectory.append(x)
            
        return torch.stack(trajectory)