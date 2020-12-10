import numpy as np
import pandas as pd

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import seaborn as sns
sns.set_palette('tab10')

import GPy
from sklearn.model_selection import KFold, RepeatedKFold

def get_data():
    df0 = pd.read_csv('./data/Data1stExpe.csv')
    df0.columns = ['user', 'condition', 'target', 'trial', 'tx', 'ty', 'px', 'py']

    df1 = pd.read_csv('./data/Data2ndExpe.csv')
    df1.columns = ['user', 'target', 'a-', 'a+']

    return df0, df1


def prepare_data(df0, df1):

    # deviation from target
    dx = df0.groupby(['target', 'condition', 'user']).apply(lambda x: x['px'].mean() - x['tx'].mean())
    dy = df0.groupby(['target', 'condition', 'user']).apply(lambda x: x['py'].mean() - x['ty'].mean())

    # dispersion from target
    vx = df0.groupby(['target', 'condition', 'user']).apply(lambda x: np.cov(x[['px', 'py']].T)[0,0])
    vy = df0.groupby(['target', 'condition', 'user']).apply(lambda x: np.cov(x[['px', 'py']].T)[1,1])
    cxy = df0.groupby(['target', 'condition', 'user']).apply(lambda x: np.cov(x[['px', 'py']].T)[0,1])

    data0 = pd.concat([dx, dy, vx, vy, cxy], axis=1)
    data0.columns=['dx', 'dy', 'vx', 'vy', 'cxy']
    data0 = data0.reset_index()

    data1 = df1.copy()
    data1['da'] = data1['a+'] - data1['a-']

    return data0, data1

def plot_data0(data0):
    fig = plt.figure(constrained_layout=True, figsize=(12,12))
    spec2 = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
    f2_ax1 = fig.add_subplot(spec2[3, 0])
    f2_ax2 = fig.add_subplot(spec2[2, 0])
    f2_ax3 = fig.add_subplot(spec2[1, 0])
    f2_ax4 = fig.add_subplot(spec2[0, 0])
    f2_ax5 = fig.add_subplot(spec2[1, 1])
    f2_ax6 = fig.add_subplot(spec2[2, 2])
    f2_ax7 = fig.add_subplot(spec2[3, 3])
    axs = [f2_ax1, f2_ax2, f2_ax3, f2_ax4, f2_ax5, f2_ax6, f2_ax7]

    for i, (j, grp) in enumerate(data0.groupby(['target', 'condition'])):
        target = j[0]
        condition = j[1]

        scatter = axs[target-1].scatter(grp['dx'], grp['dy'], label=grp['condition'].iloc[0])
        axs[target-1].scatter(0, 0, c='k', marker='x')

        color = scatter.to_rgba(0)
        confidence_ellipse(grp['dx'], grp['dy'], axs[target-1], edgecolor=default_colors[condition-1])

        axs[target-1].set_title("target {}".format(target))


    axs[0].legend()
    _=fig.suptitle('Per conditions, across targets - deviation means from target.')

    return fig, axs

def plot_data1(data1):
    fig, ax = plt.subplots()
    for i, grp in data1.groupby('target'):
        grp.plot(x='user', y='da', ax=ax, label=i)
        ax.set_title("all targets")

    return fig, ax


def get_Xy(data0, data1):
    X = data0.set_index(['target', 'user', 'condition']).unstack()
    X = X.reset_index()
    # this removes the multiindex from columns, and creates a simpler single index
    new_cols = [str(i)+str(j) for (i,j) in pd.Index(X.columns._values, tupleize_cols=False)]
    X.columns = new_cols


    # Here, we consider that measurements in experiment 1 and targets [1,2,3,4] are related to the
    # measurement of target 1 in experiment 2, while 5, 6 and 7 are linked to 2, 3 and 4, respectively.
    y = X[['target', 'user']].copy()
    y['da'] = pd.Series(dtype=float)
    y = y.set_index(['target', 'user'])
    for i, row in X.iterrows():
        user = row['user']
        target = row['target']

        target_map = {1:1, 2:1, 3:1, 4:1, 5:2, 6:3, 7:4}
        y.loc[target, user] = data1.set_index(['target', 'user']).loc[target_map[target], user]['da']
    y = y.reset_index()

    return X, y

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


from functools import reduce
from operator import and_, or_

def select(df, **kwargs):
    '''Builds a boolean array where columns indicated by keys in kwargs are tested for equality to their values.
    In the case where a value is a list, a logical or is performed between the list of resulting boolean arrays.
    Finally, a logical and is performed between all boolean arrays.
    '''

    res = []
    for k, v in kwargs.items():

        # TODO: if iterable, expand with list(iter)

        # multiple column selection with logical or
        if isinstance(v, list):
            res_or = []
            for w in v:
                res_or.append(df[k] == w)
            res_or = reduce(lambda x, y: or_(x,y), res_or)
            res.append(res_or)

        # single column selection
        else:
            res.append(df[k] == v)

    # logical and
    if res:
        res = reduce(lambda x, y: and_(x,y), res)
        res = df[res]
    else:
        res = df

    return res