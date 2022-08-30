import numpy as np
import matplotlib.pylab as plt


def make_figure_learning_across_trials(dual_model, rpe_model, ape_model, xs, show_std=True):
    fig1 = plt.figure(figsize=(8, 4))
    colors = ['gray', 'cyan', 'orange']
    plt.axhline(50, ls='dotted', alpha=0.4, color='k')
    plt.axhline(100, ls='dotted', alpha=0.4, color='k')

    for i, arr in enumerate([dual_model, rpe_model, ape_model]):
        arr_mean = np.mean(arr, axis=0)
        arr_std = np.std(arr, axis=0)
        plt.plot(xs, arr_mean, color=colors[i])
        if show_std:
            y1 = arr_mean - arr_std
            y2 = arr_mean + arr_std
            plt.fill_between(xs, y1, y2, where=y2 >= y1, color=colors[i], alpha=.2, interpolate=False)

    plt.xlabel('Trials', fontsize=20)
    plt.ylabel('Task Performance (%)', fontsize=20)
    # plt.legend(loc=(0.76, 0.3), frameon=False)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # remove the legend as the figure has it's own
    # ax.get_legend().remove()

    ax.set_xlim((0, 5000))
    ax.set_ylim((30, 101))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    # plt.title('Task learning progression')
    return fig1


def make_figure_mean_differences_quantiles(model_diffs, top_ci, bot_ci, xsbin):
    fig2 = plt.figure(figsize=(8, 4))
    plt.axhline(0, ls='dotted', alpha=0.4, color='k')
    plt.plot(xsbin, model_diffs, color='k', label='observed data')
    plt.plot(xsbin, top_ci, linestyle='--', color='gray', label='95% ci')
    plt.plot(xsbin, bot_ci, linestyle='--', color='gray')
    plt.fill_between(xsbin, bot_ci, model_diffs, where=model_diffs <= bot_ci,
                    facecolor='k', alpha=.2, interpolate=True)
    plt.ylabel('performance difference (%)')
    plt.xlabel('trial number')
    plt.legend(loc=(0.75, 0.05), frameon=False)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim((0,5000))

    return fig2


def make_figure_plot_x_weight_of_models(agents, labels, x, innact_times=False):

    fig3 = plt.figure(figsize=(8, 4))
    for i, agent in enumerate(agents):
        # correct that the last 100 original values are 0
        plt.plot(agent[x, :-1], label=labels[i])
    # innactivation points
    if innact_times:
        for xp in innact_times:
            plt.axvline(xp, ls='dotted', alpha=0.4, color='k')
    plt.legend(loc=(0.75, 0.05), frameon=False)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim((0,5000))

    return fig3


def make_figure_plot_all_weight_of_models(agents, labels):
    fig5, axs = plt.subplots(len(agents), 1, figsize=(8, 9))
    axs = axs.ravel()
    for i, agent in enumerate(agents):
        ax = axs[i]
        # correct that the last 100 original values are 0
        for k in range(4):
            ax.plot(agent[k, :-1])
        ax.set_title(labels[i])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig5


def make_figure_weights_barplot(dataforbarplot, xs):
    fig4 = plt.figure(figsize=(2, 4))
    plt.bar(xs, dataforbarplot)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig4


def make_figure_innactivation_performance(xs, dataforbarplot, cols):
    fig6 = plt.figure(figsize=(6, 4))
    plt.bar(xs, dataforbarplot, color=cols)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(50,100)

    return fig6
