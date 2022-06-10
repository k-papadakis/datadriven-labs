# %%
import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import seaborn as sns


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
                
                
def external_heat_func(x, y, mu=0.05, sigma=0.005, seed=None):
    assert np.shape(x) == np.shape(y)
    shape = np.shape(x)
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, shape)
    val = 100 * np.exp(
        -((x - 0.55)**2 + (y - 0.45)**2) / r
    )
    return val


def get_temperatures(minval, maxval, step):
    xvals = yvals = np.arange(minval+step, maxval, step)  # h, 2h, ... 1-h
    x_mesh, y_mesh = np.meshgrid(xvals, yvals)
    z = external_heat_func(x_mesh, y_mesh)
    return z.flatten(order='C')


conmat = get_connection_matrix(39, 39)
temps = get_temperatures(0, 1, 1/40)
solution = spsolve(conmat, temps)
solution = solution.reshape(39, 39)
solution = np.pad(solution, ((1, 1), (1, 1)))

ax = sns.heatmap(solution, cmap='rocket')
# ax.set_axis_off()
ax.set_title('Exact Solution')

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


