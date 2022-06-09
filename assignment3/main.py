"""
PLAN:

Submit a notebook. Justify it by saying you needed it interactive 3D plots

Field usages:
Harsh Acceleration / Break Indicators: NewRotRateX (pitch), NewAccelY
Harsh Turns: NewRotRateY (roll), NewRotRateZ (yaw), NewAccelX

3D Plot a random subset of the data (keep that subset the rest of the scatter plots)

Find the Gaussian Envelope with contamination=0.01 (proportion of outliers)
Plot the envelope https://stackoverflow.com/questions/42141505/plot-ellipse3d-in-r-plotly

(IGNORE) Find an appropriate K for K-Means using the elbow method. Find outliers.
Better to not do K means as there's probably only one cluster.
Use Local Outlier Factor instead.
3d Plot with colored clusters and marked outliers

Use an isolation forest with contamination=0.01

Find a proper time series approach (only pointwise is feasible it seems).
Maybe use only one variable, or the squared L2 norm.
Use scipy.signal?
Use this? https://analyticsindiamag.com/a-hands-on-guide-to-anomaly-detection-in-time-series-using-adtk/
https://adtk.readthedocs.io/en/stable/api/detectors.html

Say that because we can't train with normal data only,
we can't train a sequence based model like LSTM-Autoencoder,
or an adtk.detector.AutoregressionAD for outlier detection.
TRY IT ANYWAY!

Say that since the abrupt turns last very short,
we just need to detect points, not subsequences.

Store the results in CSVs, one for each approach.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import scipy.stats as st

from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


RANDOM_STATE = 7
CONTAMINATION = 0.01
DATA_PATH = 'all_LAT_uns.csv'
TURN_COLS = [
    'NewRotRateY',
    'NewRotRateZ',
    'NewAccelX',
]
    
    
df_all = pd.read_csv(DATA_PATH)
df = df_all[TURN_COLS]

rng = np.random.default_rng(RANDOM_STATE)
sample_indices = rng.choice(len(df), 5_000)
is_sample = np.full(len(df), False, bool)
is_sample[sample_indices] = True

def plot_outliers(data, preds, mask=None, title=None):
    if mask is not None:
        preds = preds[mask]
        data = data[mask]
    colors = np.where(preds == 1, 'regular', 'outlier')
    fig = px.scatter_3d(
        data, *TURN_COLS,
        color=colors,
        color_discrete_sequence=['blue', 'red'],
        title=f'{title} ({len(data)} samples)'
    )
    fig.show()

# %%
# ELLIPTIC ENVELOPE

ee = EllipticEnvelope(contamination=CONTAMINATION, random_state=RANDOM_STATE)
ee_preds =ee.fit_predict(df)

plot_outliers(
    df, ee_preds, mask=is_sample,
    title=f'Gaussian Outlier Detection (contamination={ee.contamination})',
)

# %%
# LOCAL OUTLIER FACTOR
lof = LocalOutlierFactor(n_neighbors=4, contamination=CONTAMINATION)
lof_preds = lof.fit_predict(df)

plot_outliers(
    df, lof_preds, mask=None,
    title=f'LOF (contamination={lof.contamination})'
)
# %%
# ISOLATION FOREST
isfo = IsolationForest(n_estimators=100, contamination=CONTAMINATION)
isfo_preds = isfo.fit_predict(df)

plot_outliers(
    df, isfo_preds, mask=is_sample,
    title=f'Isolation Forest (contamination={isfo.contamination})'
)

# %%
