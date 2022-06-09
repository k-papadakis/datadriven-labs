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

from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from adtk.detector import AutoregressionAD
from adtk.visualization import plot as adtkplot


RANDOM_STATE = 7
CONTAMINATION = 0.01
DATA_PATH = 'all_LAT_uns.csv'
TURN_COLS = [
    'NewRotRateY',
    'NewRotRateZ',
    'NewAccelX',
]


# %%    
# Load and plot the data
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
    
fig = px.scatter_3d(df[is_sample], *TURN_COLS)
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
lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
lof_preds = lof.fit_predict(df)

plot_outliers(
    df, lof_preds, mask=is_sample,
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
norms = (df ** 2).sum(axis=1) ** 0.5
time_index = pd.date_range(start='2021', periods=len(norms), freq='1S')
norms = norms.set_axis(time_index)
norms.name = 'sqrt(' + ' + '.join(f'{col} ^ 2' for col in TURN_COLS) + ')'

# %%
# Auto-Regression
# c is used to compute the interval where the regular values lie:
# [Q1−c*IQR, Q3+c*IQR] where IQR=Q3−Q1 and Q1, Q3 the the 25% and 75% quantiles]
c = 12.5  # results to about 0.01 contamination
arad = AutoregressionAD(n_steps=10, step_size=1, c=c, side='both')
arad_preds = arad.fit_detect(norms)

adtkplot(norms, anomaly=arad_preds, anomaly_color="red", anomaly_tag="marker")

# %%
# Find Overlap
preds = pd.DataFrame.from_dict({
    'Eliptic Envelope': ee_preds,
    'Local Outlier Factor': lof_preds,
    'Isolation Forest': isfo_preds,
    'Auto Regression': np.where(arad_preds.fillna(False), -1, 1),
}, dtype=int)

res = np.empty((preds.shape[1], preds.shape[1]), int)
for i, col1 in enumerate(preds.columns):
    for j, col2 in enumerate(preds.columns):
        n_same_outliers = (preds[col1] + preds[col2] == -2).sum()
        res[i, j] = n_same_outliers

res = pd.DataFrame(res, index=preds.columns, columns=preds.columns, dtype=int)

ax = sns.heatmap(res, annot=True, fmt='d', cmap='viridis')
ax.set_title('Common Predicted Outliers')

# %%
# EXPORT RESULTS
preds.to_csv('predictions.csv', index=False)
