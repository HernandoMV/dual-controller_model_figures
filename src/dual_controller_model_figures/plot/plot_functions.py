import numpy as np
import pandas as pd
from os import path
import matplotlib.pylab as plt

def make_figure_learning_across_trials(dual_model, rpe_model, ape_model):
    fig1 = plt.figure(figsize=(8, 4))
    xs = range(0, dual_model.shape[1]*10, 10)
    colors = ['gray', 'cyan', 'orange']
    plt.axhline(50, ls='dotted', alpha=0.4, color='k')
    plt.axhline(100, ls='dotted', alpha=0.4, color='k')
    for i, arr in enumerate([dual_model, rpe_model, ape_model]):
        arr_mean = 100 * np.mean(arr, axis=0)
        arr_std = 100 * np.std(arr, axis=0)
        plt.plot(xs, arr_mean, color=colors[i])
        y1 = arr_mean - arr_std
        y2 = arr_mean + arr_std
        # plt.fill_between(xs, y1, y2, where=y2 >= y1, color=colors[i], alpha=.2, interpolate=False)

    plt.xlabel('Trials', fontsize=20)
    plt.ylabel('Task Performance (%)', fontsize=20)
    plt.legend(loc=(0.76, 0.3), frameon=False)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # remove the legend as the figure has it's own
    # ax.get_legend().remove()

    ax.set_xlim((0, 5000))
    ax.set_ylim((30, 100))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    # plt.title('Task learning progression')
    return fig1
