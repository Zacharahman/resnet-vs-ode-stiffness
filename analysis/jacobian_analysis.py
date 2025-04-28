import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvals
from data.generate_stiff_systems import generate_van_der_pol, generate_lotka_volterra

def compute_jacobian_vdp(x, mu=1000.0):
    """Compute Jacobian matrix for Van der Pol oscillator."""
    x1, x2 = x
    J = np.array([
        [0, 1],
        [-1 - 2*mu*x1*x2, mu*(1 - x1**2)]
    ])
    return J

def compute_jacobian_lv(x, alpha=1.5, beta=1.0, delta=0.75, gamma=1.0):
    """Compute Jacobian matrix for Lotka-Volterra system."""
    x1, x2 = x
    J = np.array([
        [alpha - beta*x2, -beta*x1],
        [delta*x2, delta*x1 - gamma]
    ])
    return J

def analyze_stiffness(t, y, compute_jacobian, title):
    """Analyze stiffness along a trajectory."""
    # Calculate eigenvalues at each point
    eigenvalues = np.array([eigvals(compute_jacobian(y[i])) for i in range(len(y))])
    
    # Calculate stiffness ratio (ratio of largest to smallest eigenvalue magnitudes)
    stiffness_ratio = np.abs(eigenvalues).max(axis=1) / np.abs(eigenvalues).min(axis=1)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot eigenvalue trajectories
    ax1.plot(t, eigenvalues.real, 'b.', label='Real part', alpha=0.5)
    ax1.plot(t, eigenvalues.imag, 'r.', label='Imaginary part', alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Eigenvalue components')
    ax1.set_title(f'Eigenvalue Evolution - {title}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot stiffness ratio
    ax2.semilogy(t, stiffness_ratio)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Stiffness Ratio (log scale)')
    ax2.set_title(f'Stiffness Ratio Evolution - {title}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/stiffness_analysis_{title.lower().replace(" ", "_")}.png')
    plt.close()
    
    return np.mean(stiffness_ratio), np.max(stiffness_ratio)

def plot_stability_regions():
    """Plot stability regions for different numerical methods."""
    # Create complex plane grid
    x = np.linspace(-5, 1, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Forward Euler stability region: |1 + z| <= 1
    euler_stable = np.abs(1 + Z) <= 1
    ax1.contourf(X, Y, euler_stable, levels=[0, 0.5, 1], cmap='RdYlBu')
    ax1.set_title('Forward Euler Stability Region')
    ax1.set_xlabel('Re(λh)')
    ax1.set_ylabel('Im(λh)')
    ax1.grid(True)
    
    # BDF2 stability region
    def bdf2_stability(z):
        return np.abs((2/3 + z)/(1 - z/3))
    
    bdf2_stable = bdf2_stability(Z) <= 1
    ax2.contourf(X, Y, bdf2_stable, levels=[0, 0.5, 1], cmap='RdYlBu')
    ax2.set_title('BDF2 Stability Region')
    ax2.set_xlabel('Re(λh)')
    ax2.set_ylabel('Im(λh)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/stability_regions.png')
    plt.close()

if __name__ == '__main__':
    # Generate trajectories
    t_vdp, y_vdp = generate_van_der_pol()
    t_lv, y_lv = generate_lotka_volterra()
    
    # Analyze stiffness
    print('\nAnalyzing Van der Pol oscillator...')
    mean_stiff_vdp, max_stiff_vdp = analyze_stiffness(
        t_vdp, y_vdp, compute_jacobian_vdp, 'Van der Pol'
    )
    print(f'Mean stiffness ratio: {mean_stiff_vdp:.2f}')
    print(f'Max stiffness ratio: {max_stiff_vdp:.2f}')
    
    print('\nAnalyzing Lotka-Volterra system...')
    mean_stiff_lv, max_stiff_lv = analyze_stiffness(
        t_lv, y_lv, compute_jacobian_lv, 'Lotka-Volterra'
    )
    print(f'Mean stiffness ratio: {mean_stiff_lv:.2f}')
    print(f'Max stiffness ratio: {max_stiff_lv:.2f}')
    
    # Plot stability regions
    print('\nPlotting stability regions...')
    plot_stability_regions()
    print('\nAnalysis complete! Check results/plots/ for visualizations.')