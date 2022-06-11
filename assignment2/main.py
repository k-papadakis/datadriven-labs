# %%
import numpy as np
from scipy.sparse import lil_array, issparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve, svd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42


# %%
def get_connection_matrix(
    m, n,
    self_weight=4.0 * 40**2,
    other_weight=-1.0 * 40**2,
    directions=((0,1), (0, -1), (1, 0), (-1, 0)),
):
    conmat = lil_array((m*n, m*n), dtype=np.float64)
    conmat.setdiag(self_weight)
    for i in range(m):
        for j in range(n):
            for a, b in directions:
                k, l = i+a, j+b
                if 0 <= k < m and 0 <= l < n:
                    x = i*n + j
                    y = k*n + l
                    conmat[x, y] = other_weight
    return conmat.tocsr()
                
                
def external_heat_func(x, y, mu, sigma, seed=None):
    assert np.shape(x) == np.shape(y)
    shape = np.shape(x)
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, shape)
    val = 100 * np.exp(
        -((x - 0.55)**2 + (y - 0.45)**2) / r
    )
    return val


def get_temperatures(minval, maxval, step, mu=0.05, sigma=0.005, seed=None):
    xvals = yvals = np.arange(minval+step, maxval, step)  # h, 2h, ... 1-h
    x_mesh, y_mesh = np.meshgrid(xvals, yvals)
    z = external_heat_func(x_mesh, y_mesh, mu, sigma, seed=seed)
    return z.flatten(order='C')


def get_pca(X):
    n, p = X.shape
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    U, S, Vt = svd(X, full_matrices=False)
    eigvals = S**2 / (n - 1)
    eigvecs = Vt.T
    return eigvals, eigvecs


def solve_system(conmat, temps, pca: None | np.ndarray = None):
    if pca is not None:
        conmat = pca.T @ conmat @ pca
        temps = pca.T @ temps
    
    if issparse(conmat):
        solution = spsolve(conmat, temps)
    else:
        solution = solve(conmat, temps)
        
    if pca is not None:
        solution = pca @ solution
    
    solution = solution.T
    assert solution.shape[-1] == 39*39
    solution = solution.reshape(-1, 39, 39)
    solution = np.pad(solution, ((0, 0), (1, 1), (1, 1)))
    
    if solution.shape[0] == 1:
        solution = np.squeeze(solution, 0)
        
    return solution


def monte_carlo(n_samples, conmat, pca: None | np.ndarray = None, seed=None):
    seeds = range(seed, seed + n_samples) if seed is not None else [None] * n_samples
    
    temps = []
    for i in range(n_samples):
        t = get_temperatures(0, 1, 1/40, seed=seeds[i])
        temps.append(t)
    temps = np.array(temps)
    
    solutions = solve_system(conmat, temps.T, pca=pca)
    return solutions


def plot_solution(temps, solution, title=None):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    
    sns.heatmap(
        np.pad(temps.reshape(39, 39), ((1,1), (1, 1))),
        cmap='rocket', square=True,
        vmin=0, vmax=np.max(temps),
        ax=axs[0]
    )
    sns.heatmap(
        solution,
        cmap='rocket', square=True,
        vmin=0,
        vmax=2 / (1/np.max(solution) + 1/np.max(temps)),
        ax=axs[1]
    )
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    axs[0].set_title('Heat Source')
    axs[1].set_title('Equilibrium')
    fig.suptitle(title)
    
    return axs


def plot_explained_variance(ratio_cumsum, n_components):
    point = n_components, ratio_cumsum[n_components - 1]
    xs = range(1, 1 + len(ratio_cumsum))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, ratio_cumsum)
    ax.plot(*point, 'ro', label=f'({point[0]+1: d}, {point[1]: .2f})')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.legend(loc='best')
    ax.set_title('PCA')
    
    return ax


def plot_density(samples, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    
    sns.histplot(samples, kde=True, stat='density', ax=ax)
    ax.set_title(
        ('' if title is None else title + ' ')
        + f'N={len(samples):,}, '
        f'mean={samples.mean():.2f}, '
        f'stdev={samples.std():.2f}'
    )
    return ax


def plot_exact_vs_pca(samples_exact, samples_pca):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    plot_density(samples_exact, title='Exact Solutions', ax=axs[0])
    plot_density(samples_pca, title=f'PCA Solutions', ax=axs[1])
    fig.suptitle('Monte Carlo simulation of the heat equilibrium at the center.')
    return axs


# %%
# FIND THE SOLUTION OF ONE INSTANCE
conmat = get_connection_matrix(39, 39)
t = get_temperatures(0, 1, 1/40, seed=RANDOM_STATE)
solution = solve_system(conmat, t)
plot_solution(t, solution, title='Exact Solution')
plt.savefig('output/solution-heatmap.png', facecolor='white', transparent=False)


# %%
# MONTE CARLO (EXACT)
# assert n_samples > 39*39 = 1521
samples = monte_carlo(20_000, conmat, seed=RANDOM_STATE)
c1 = (samples.shape[-2] + 1) // 2
c2 = (samples.shape[-1] + 1) // 2
center_samples = samples[..., c1, c2]
plot_density(
    center_samples,
    title='Monte Carlo simulation of the heat equilibrium at the center.\n'
)
plt.savefig('output/dist-exact.png', facecolor='white', transparent=False)


# %%
# PCA
samples_flat = samples[:, 1:-1, 1:-1]
samples_flat = samples_flat.reshape(samples.shape[0], -1)

eigvals, eigvecs = get_pca(samples_flat)
explained_var = eigvals / np.sum(eigvals)
ratio_cumsum = np.cumsum(explained_var)
thresh = 0.99
n_components = 1 + np.searchsorted(ratio_cumsum, thresh, side="right")
plot_explained_variance(ratio_cumsum, n_components)
plt.savefig('output/explained-var.png', facecolor='white', transparent=False)

# %%
# MONTE CARLO (PCA)
samples_alt = monte_carlo(20_000, conmat, pca=eigvecs[:, :n_components], seed=RANDOM_STATE)
center_samples_pca = samples_alt[:, c1, c2]
plot_exact_vs_pca(center_samples, center_samples_pca)
plt.savefig('output/dists-exact-pca.png', facecolor='white', transparent=False)

# %%
plt.show()

# %%
# Normality test
# H0: "The data are normally distributed".
# The p-value is very large and thus H0 is not rejected. 
print(f'Normality Test for the Exact Monte Carlo:\n{st.normaltest(center_samples)}')

# %%