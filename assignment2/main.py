# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
def get_connection_matrix(
    m, n,
    self_weight=4.0,
    other_weight=-1.0,
    directions=((0,1), (0, -1), (1, 0), (-1, 0)),
):
    conmat = np.zeros((m*n, m*n), float)
    np.fill_diagonal(conmat, self_weight)
    for i in range(m):
        for j in range(n):
            for a, b in directions:
                k, l = i+a, j+b
                if 0 <= k < m and 0 <= l < n:
                    x = i*n + j
                    y = k*n + l
                    conmat[x, y] = other_weight
                    
    return conmat
                

conmat = get_connection_matrix(39, 39)

