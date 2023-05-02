import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def plot_ts(y, img_name="series", length=10, title=None):
    n = y.shape[0]
    l = [i for i in range(1, length+1)]
    acf_values = acf(y, nlags=length, fft=False)
    pacf_values = pacf(y, nlags=length)

    df = pd.DataFrame({'t': l, 'acf': acf_values[1:], 'pacf': pacf_values[1:]})

    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[2, 1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    if title is None:
        ax0.plot(np.arange(1, n+1), y, label=None)
    else:
        ax0.plot(np.arange(1, n+1), y, label=None)
        ax0.set_title(title)

    ax1.bar(df.t, df.acf)
    ax1.set_title("AutoCorrelation")
    ax2.bar(df.t, df.pacf)
    ax2.set_title("Partial AutoCorrelation")

    plt.tight_layout()
    plt.savefig(img_name)
    plt.show()
