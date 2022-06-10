# %%
import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import spsolve
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_SEED = 42

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


def get_temperatures(minval, maxval, step,mu=0.05, sigma=0.005, seed=None):
    xvals = yvals = np.arange(minval+step, maxval, step)  # h, 2h, ... 1-h
    x_mesh, y_mesh = np.meshgrid(xvals, yvals)
    z = external_heat_func(x_mesh, y_mesh, mu, sigma, seed=seed)
    return z.flatten(order='C')


def exact_solve_system(conmat, seed=None):
    temps = get_temperatures(0, 1, 1/40, seed=seed)
    solution = spsolve(conmat, temps)
    solution = solution.reshape(39, 39)
    solution = np.pad(solution, ((1, 1), (1, 1)))
    return solution


conmat = get_connection_matrix(39, 39)
solution = exact_solve_system(conmat, seed=RANDOM_SEED)
ax = sns.heatmap(solution, cmap='rocket')
ax.invert_yaxis()
ax.set_title('Exact Solution')
plt.savefig('heatmap.png', facecolor='white', transparent=False)

# # Compare sparse vs dense times
# import timeit
# from scipy.linalg import solve
# dense_conmat = conmat.todense()
# sparse_time = timeit.timeit(lambda: spsolve(conmat, temps), number=100)
# dense_time = timeit.timeit(lambda: solve(dense_conmat, temps), number=100)
# print(f'Sparse Time: {sparse_time}\nDense Time: {dense_time}')
# # 0.27s
# # 8.93s


# %%
n_mc_samples = 10_000

mc_samples = np.empty(n_mc_samples)
for i in range(n_mc_samples):
    sol = exact_solve_system(conmat)
    # c1, c2 = (sol.shape[0] + 1) // 2, (sol.shape[1] + 1) // 2
    mc_samples[i] = sol[21, 21]

# %%
mc_mean = mc_samples.mean()
mc_std = mc_samples.std()

ax = sns.histplot(mc_samples, kde=True, stat='density')
ax.set_title('MC for heat at (0.5, 0.5). '
             f'N={n_mc_samples:,}, mean={mc_mean:.2f}, stdev={mc_std:.2f}')
# plt.savefig('montecarlo.png', facecolor='white', transparent=False)


# %%