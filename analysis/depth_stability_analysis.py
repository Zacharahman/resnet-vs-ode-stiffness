import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import svd
from models.resnet import ResNet
from models.neural_ode import NeuralODE
from data.generate_stiff_systems import generate_van_der_pol, generate_lotka_volterra

def compute_network_jacobian(model, x):
    """Compute Jacobian of network output with respect to input using autograd."""
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    y = model(x)
    
    # Initialize Jacobian matrix
    jac = torch.zeros(y.shape[0], x.shape[0])
    
    # Compute Jacobian row by row
    for i in range(y.shape[0]):
        model.zero_grad()
        y[i].backward(retain_graph=True)
        jac[i] = x.grad
        x.grad.zero_()
    
    return jac.detach().numpy()

def analyze_depth_stability(model_type='resnet', system='van_der_pol', num_points=10):
    """Analyze stability at different depths/timesteps."""
    # Load data
    if system == 'van_der_pol':
        t, y = generate_van_der_pol()
    else:
        t, y = generate_lotka_volterra()
    
    state_dim = y.shape[1]
    
    # Initialize models with different depths/integration times
    depths = [2, 4, 8, 16, 32]  # For ResNet: number of layers, For NODE: relative integration time
    singular_values = []
    condition_numbers = []
    
    for depth in depths:
        if model_type == 'resnet':
            model = ResNet(state_dim=state_dim, num_layers=depth)
        else:  # neural_ode
            model = NeuralODE(state_dim=state_dim)
        
        # Sample points from trajectory
        indices = np.linspace(0, len(y)-1, num_points, dtype=int)
        points = y[indices]
        
        # Compute Jacobian and its singular values at each point
        depth_singvals = []
        for point in points:
            jac = compute_network_jacobian(model, point)
            _, s, _ = svd(jac)
            depth_singvals.append(s)
        
        # Average over points
        avg_singvals = np.mean(depth_singvals, axis=0)
        singular_values.append(avg_singvals)
        condition_numbers.append(avg_singvals[0] / avg_singvals[-1])
    
    return depths, singular_values, condition_numbers

def plot_stability_comparison(system='van_der_pol'):
    """Compare stability characteristics of ResNet vs Neural ODE."""
    # Analyze both models
    depths_resnet, singvals_resnet, cond_resnet = analyze_depth_stability('resnet', system)
    depths_node, singvals_node, cond_node = analyze_depth_stability('neural_ode', system)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot singular value distributions
    for i, depth in enumerate(depths_resnet):
        ax1.semilogy(range(len(singvals_resnet[i])), singvals_resnet[i], 
                    'b-', alpha=0.3, label=f'ResNet (depth={depth})')
        ax1.semilogy(range(len(singvals_node[i])), singvals_node[i], 
                    'r-', alpha=0.3, label=f'NODE (T={depth})')
    
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Magnitude (log scale)')
    ax1.set_title(f'Singular Value Distribution\n{system.replace("_", " ").title()}')
    ax1.grid(True)
    
    # Plot condition number growth
    ax2.semilogy(depths_resnet, cond_resnet, 'b.-', label='ResNet')
    ax2.semilogy(depths_node, cond_node, 'r.-', label='Neural ODE')
    ax2.set_xlabel('Depth / Integration Time')
    ax2.set_ylabel('Condition Number (log scale)')
    ax2.set_title(f'Stability Degradation with Depth\n{system.replace("_", " ").title()}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/plots/depth_stability_{system}.png')
    plt.close()

if __name__ == '__main__':
    print("\nAnalyzing Van der Pol system...")
    plot_stability_comparison('van_der_pol')
    
    print("Analyzing Lotka-Volterra system...")
    plot_stability_comparison('lotka_volterra')
    
    print("\nAnalysis complete! Check results/plots/ for visualizations.")
