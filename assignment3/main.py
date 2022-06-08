# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

from sklearn.cluster import KMeans

# %%
df = pd.read_csv('all_LAT_uns.csv')
xcol = 'NewRotRateX'
ycol = 'NewRotRateY'
# %%
rot = df[[xcol, ycol]]
rot_norm = np.sum(rot**2, axis=1)
plt.plot(rot_norm)
plt.title('Rotation Rate L2 Norm Squared')

# %%
# models = [KMeans(n_clusters=i) for i in range(2, 10)]
# for model in models:
#     model.fit(df)
# scores = [model.inertia_ for model in models]
# plt.plot(scores)


# %%
mu = np.mean(rot, 0)
covmat = np.cov(rot.T)
distr = st.multivariate_normal(mu, covmat)
threshold = 1e-16
densities = distr.pdf(rot)

is_outlier = densities < threshold
n_outliers = is_outlier.sum()
n_regulars = len(is_outlier) - n_outliers

ax = df[~is_outlier].plot.scatter(xcol, ycol, c='blue', label='regular', figsize=(8, 8))
df[is_outlier].plot.scatter(xcol, ycol, c='red', label='outlier', ax=ax)
ax.set_aspect('equal', 'box')
ax.set_title(f'#Regulars={n_regulars:,}, #Outliers={n_outliers:,}')
plt.legend()
# %%
