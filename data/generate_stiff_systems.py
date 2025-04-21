import os
import numpy as np
from scipy.integrate import solve_ivp


def generate_van_der_pol(mu=1000.0,
                          t_span=(0.0, 20.0),
                          y0=None,
                          num_points=1000,
                          method='BDF'):
    """
    Simulate the Van der Pol oscillator in the stiff regime.
    """
    if y0 is None:
        y0 = np.array([2.0, 0.0])
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    def vdp(t, y):
        x, v = y
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x
        return [dxdt, dvdt]

    sol = solve_ivp(vdp, t_span, y0, method=method, t_eval=t_eval)
    return sol.t, sol.y.T


def generate_lotka_volterra(alpha=1.5,
                            beta=1.0,
                            delta=0.75,
                            gamma=1.0,
                            t_span=(0.0, 15.0),
                            y0=None,
                            num_points=1000,
                            method='BDF'):
    """
    Simulate the Lotka-Volterra predator-prey system.
    """
    if y0 is None:
        y0 = np.array([10.0, 5.0])
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    def lv(t, y):
        x, y_pred = y
        dxdt = alpha * x - beta * x * y_pred
        dydt = delta * x * y_pred - gamma * y_pred
        return [dxdt, dydt]

    sol = solve_ivp(lv, t_span, y0, method=method, t_eval=t_eval)
    return sol.t, sol.y.T


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)

    # Generate and save datasets
    t_vdp, y_vdp = generate_van_der_pol()
    t_lv, y_lv = generate_lotka_volterra()

    np.savez(os.path.join(data_dir, "van_der_pol.npz"), t=t_vdp, y=y_vdp)
    np.savez(os.path.join(data_dir, "lotka_volterra.npz"), t=t_lv, y=y_lv)

    print("✅ Generated and saved van_der_pol.npz and lotka_volterra.npz in data/")
