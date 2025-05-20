# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D

def plot_dirichlet_3d(alpha):
    # Create simplex coordinates: a triangle grid in 2D for 3D Dirichlet
    resolution = 200
    x = np.linspace(0.001, 0.999, resolution)
    y = np.linspace(0.001, 0.999, resolution)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y

    mask = (Z > 0)  # only points where all components > 0 (inside simplex)

    # Evaluate the Dirichlet PDF
    coords = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    pdf_vals = dirichlet.pdf(coords.T, alpha)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X[mask], Y[mask], pdf_vals, cmap='viridis', linewidth=0.2)
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("Density")
    ax.set_title(f"Dirichlet PDF Surface (α={alpha})")
    plt.tight_layout()
    plt.show()


#plot_dirichlet_3d(alpha=[1.0, 1.0, 1.0])   # Uniform Dirichlet
plot_dirichlet_3d(alpha=[1.1, 2.1, 5.1])   # Skewed
#plot_dirichlet_3d(alpha=[10.0, 10.0, 10.0]) # Concentrated near center

# %%
