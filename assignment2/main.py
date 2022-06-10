# %%
import numpy as np
from scipy.sparse import lil_array, csr_array, issparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


RANDOM_STATE = 42

# %%
def get_connection_matrix(
    m, n,
    self_weight=4.0,
    other_weight=-1.0,
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


def solve_system(conmat, temps, pca=None):
    if temps.ndim == 1:
        temps = np.expand_dims(temps, 0)
    
    if pca is not None:
        temps = pca.transform(temps)
        
    if issparse(conmat):
        solution = spsolve(conmat, temps.T)
    else:
        solution = solve(conmat, temps.T)
    solution = np.transpose(solution)
        
    if pca is not None:
        solution = pca.inverse_transform(solution)
    
    assert solution.shape[-1] == 39*39
    solution = solution.reshape(-1, 39, 39)
    solution = np.pad(solution, ((0, 0), (1, 1), (1, 1)))
    
    if solution.shape[0] == 1:
        solution = np.squeeze(solution, 0)
        
    return solution


# %%
conmat = get_connection_matrix(39, 39)
ts = get_temperatures(0, 1, 1/40, seed=RANDOM_STATE)
solution = solve_system(conmat, ts)
ax = sns.heatmap(solution, cmap='rocket')
ax.invert_yaxis()
ax.set_title('Exact Solution')
# plt.savefig('heatmap.png', facecolor='white', transparent=False)

# # Compare sparse vs dense times
# import timeit
# dense_conmat = conmat.todense()
# sparse_time = timeit.timeit(lambda: spsolve(conmat, temps), number=100)
# dense_time = timeit.timeit(lambda: solve(dense_conmat, temps), number=100)
# print(f'Sparse Time: {sparse_time}\nDense Time: {dense_time}')
# # 0.27s
# # 8.93s


# %%
# MONTE CARLO
def monte_carlo(n_samples, conmat, pca=None, seed=None):
    seeds = range(seed, seed+n_samples) if seed is not None else [None]*n_samples
    
    temps = []
    for i in range(n_samples):
        t = get_temperatures(0, 1, 1/40, seed=seeds[i])
        temps.append(t)
    temps = np.array(temps)
    
    solutions = solve_system(conmat, temps, pca=pca)
    return solutions


# assert n_samples > 39*39 = 1521
samples = monte_carlo(10_000, conmat, seed=RANDOM_STATE)

# %%
c1 = (samples.shape[-2] + 1) // 2
c2 = (samples.shape[-1] + 1) // 2
center_samples = samples[:, c1, c2]

# %%
ax = sns.histplot(center_samples, kde=True, stat='density')
ax.set_title('MC for heat at (0.5, 0.5). '
             f'N={len(center_samples):,}, mean={center_samples.mean():.2f}, stdev={center_samples.std():.2f}')
# plt.savefig('montecarlo.png', facecolor='white', transparent=False)

# H0: "The data are normally distributed".
# The p-value is very large and thus H0 is not rejected. 
print(st.normaltest(center_samples))


# %%
samples_flat = samples[:, 1:-1, 1:-1]
samples_flat = samples_flat.reshape(samples.shape[0], -1)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    # ('pca', PCA(n_components=0.96)),
    ('pca', PCA()),
])
pipe.fit(samples_flat)

print(f"Number of components: {pipe['pca'].n_components_}")
print(f"Explained variance ratio: {pipe['pca'].explained_variance_ratio_.sum(): .2%}")
# plt.plot(np.cumsum(pipe['pca'].explained_variance_ratio_))

# %%
# pipe.transform(samples_flat)
components = pipe['pca'].components_
conmat_red = components @ conmat @ components.T

# %%
samples_alt = monte_carlo(10_000, conmat_red, pca=pipe, seed=RANDOM_STATE)

# %%
center_samples_alt = samples_alt[:, c1, c2]
fig, axs = plt.subplots(nrows=2)
sns.histplot(center_samples, kde=True, stat='density', ax=axs[0])
sns.histplot(center_samples_alt, kde=True, stat='density', ax=axs[1])

# %%
