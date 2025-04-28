import torch
import torch.nn as nn

class AdaptiveResNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, num_layers=4, init_dt=0.1):
        super().__init__()
        self.state_dim = state_dim
        
        # Network that approximates the vector field f(t,x)
        self.f = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Learnable step sizes for each layer
        self.log_dt = nn.Parameter(torch.ones(num_layers) * torch.log(torch.tensor(init_dt)))
        self.num_layers = num_layers
    
    def forward(self, x0):
        """Forward pass using Euler discretization with learned step sizes."""
        x = x0
        
        # Apply num_layers Euler steps with learned dt
        for i in range(self.num_layers):
            # Get adaptive step size through exp to ensure positivity
            dt = torch.exp(self.log_dt[i])
            
            # Euler step: x_{n+1} = x_n + dt * f(x_n)
            dx = self.f(x)
            x = x + dt * dx
            
        return x
    
    def trajectory(self, x0, num_steps):
        """Generate trajectory by applying forward pass multiple times."""
        trajectory = [x0]
        x = x0
        
        for _ in range(num_steps):
            x = self.forward(x)
            trajectory.append(x)
            
        return torch.stack(trajectory)
    
    def get_step_sizes(self):
        """Return the current learned step sizes."""
        return torch.exp(self.log_dt).detach()