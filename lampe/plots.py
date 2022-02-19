r"""Plotting routines"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as si

from numpy import ndarray as Array
from numpy.typing import ArrayLike
from typing import *


plt.rcParams.update({
    'axes.axisbelow': True,
    'axes.linewidth': .8,
    'figure.autolayout': True,
    'figure.dpi': 150,
    'figure.figsize': (6.4, 4.8),
    'font.size': 12.,
    'legend.fontsize': 'x-small',
    'lines.linewidth': 1.,
    'lines.markersize': 3.,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'xtick.labelsize': 'x-small',
    'xtick.major.width': .8,
    'ytick.labelsize': 'x-small',
    'ytick.major.width': .8,
})

if mpl.checkdep_usetex(True):
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
    })


class LinearAlphaColormap(mpl.colors.LinearSegmentedColormap):
    r"""Linear transparency colormap segmented between levels"""

    def __new__(
        self,
        color: Union[str, tuple],
        levels: ArrayLike = None,
        alpha: Tuple[float, float] = (0., 1.),
        name: str = None,
        **kwargs,
    ):
        if name is None:
            if type(color) is str:
                name = f'alpha_{color}'
            else:
                name = f'alpha_{hash(color)}'

        if levels is None:
            levels = [0., 1.]

        levels = np.asarray(levels)
        levels = (levels - levels.min()) / (levels.max() - levels.min())
        alpha = np.linspace(*alpha, len(levels))

        rgb = mpl.colors.to_rgb(color)

        colors = sorted([
            (l, rgb + (a,))
            for l, a in zip(levels, alpha)
        ])

        return mpl.colors.LinearSegmentedColormap.from_list(
            name=name,
            colors=colors,
        )


def credible_levels(hist: Array, quantiles: Array) -> Array:
    r"""Retrieve credible region boundary levels from an histogram"""

    x = np.sort(hist, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, quantiles * cdf[-1])

    return x[idx]


def corner(
    data: ArrayLike,  # table or matrix of 1d/2d histograms
    bins: Union[int, List[int]] = 100,
    bounds: Tuple[ArrayLike, ArrayLike] = None,
    quantiles: ArrayLike = [.6827, .9545, .9973],
    color: Union[str, tuple] = None,
    alpha: float = .5,
    legend: str = None,
    labels: List[str] = None,
    markers: List[ArrayLike] = [],
    smooth: float = 0,
    figure: mpl.figure.Figure = None,
    **kwargs,
) -> mpl.figure.Figure:
    r"""Corner plot"""

    # Histograms
    data = np.asarray(data)

    if np.isscalar(data[0, 0]):
        D = data.shape[-1]
        data = data.reshape(-1, D)

        if type(bins) is int:
            bins = [bins] * D

        if bounds is None:
            lower, upper = data.min(axis=0), data.max(axis=0)
        else:
            lower, upper = map(np.asarray, bounds)

        bins = [
            np.histogram_bin_edges(data, bins[i], range=(lower[i], upper[i]))
            for i in range(D)
        ]

        hists = np.ndarray((D, D), dtype=object)

        for i in range(D):
            for j in range(i + 1):
                if i == j:
                    hist, _ = np.histogram(
                        data[..., i],
                        bins=bins[i],
                        density=True,
                    )
                else:
                    hist, _, _ = np.histogram2d(
                        data[..., i], data[..., j],
                        bins=(bins[i], bins[j]),
                        density=True,
                    )

                hists[i, j] = hist
    else:
        D = len(data)

        if bounds is None:
            lower, upper = np.zeros(D), np.ones(D)
        else:
            lower, upper = map(np.asarray, bounds)

        bins = [None] * D
        for i in range(D):
            if data[i, i] is not None:
                bins[i] = np.linspace(lower[i], upper[i], len(data[i, i]) + 1)

        hists = data

    # Figure
    if figure is None:
        kwargs.setdefault('figsize', (6.4, 6.4))

        figure, axes = plt.subplots(
            D, D,
            squeeze=False,
            sharex='col',
            gridspec_kw={'wspace': 0., 'hspace': 0.},
            **kwargs,
        )
        new = True
    else:
        axes = np.asarray(figure.axes).reshape(D, D)
        new = False

    # Legend

    ## Color
    lines = axes[0, -1].plot([], [], color=color, label=legend)
    color = lines[-1].get_color()

    handles, texts = axes[0, -1].get_legend_handles_labels()

    ## Quantiles
    quantiles = np.sort(np.asarray(quantiles))[::-1]
    quantiles = np.append(quantiles, 0)

    cmap = LinearAlphaColormap('black', levels=quantiles, alpha=(0, alpha))

    levels = (quantiles[1:] + quantiles[:-1]) / 2
    levels = (levels - quantiles.min()) / (quantiles.max() - quantiles.min())

    for q, l in zip(quantiles[:-1], levels):
        handles.append(mpl.patches.Patch(color=cmap(l), linewidth=0))
        texts.append(r'{:.1f}\,\%'.format(q * 100))

    ## Update
    if not new:
        figure.legends.clear()

    figure.legend(handles, texts, loc='upper right', bbox_to_anchor=(0.975, 0.975), frameon=False)

    # Plot
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            hist = hists[i, j]

            if j > i:
                ax.axis('off')
                continue

            if hist is None:
                continue

            if smooth > 0:
                hist = si.gaussian_filter(hist, smooth)

            ## Draw
            x, y = bins[j], bins[i]
            x = (x[1:] + x[:-1]) / 2
            y = (y[1:] + y[:-1]) / 2

            if i == j:
                ax.plot(x, hist, color=color)

                _, top = ax.get_ylim()
                bottom = 0.
                top = max(top, hist.max() * 1.0625)

                ax.set_xlim(left=bins[i][0], right=bins[i][-1])
                ax.set_ylim(bottom=bottom, top=top)
            else:
                levels = np.unique(credible_levels(hist, quantiles))

                cf = ax.contourf(
                    x, y, hist,
                    levels=levels,
                    cmap=LinearAlphaColormap(color, levels, alpha=(0, alpha)),
                )
                ax.contour(cf, colors=color)

                if j > 0:
                    ax.sharey(axes[i, j - 1])
                else:
                    ax.set_ylim(bottom=bins[i][0], top=bins[i][-1])

            ## Markers
            for marker in map(np.asarray, markers):
                ax.axvline(
                    marker[j],
                    color='k',
                    linestyle='--',
                    zorder=420,
                )

                if i != j:
                    ax.axhline(
                        marker[i],
                        color='k',
                        linestyle='--',
                        zorder=420,
                    )

                    ax.plot(
                        marker[j], marker[i],
                        color='k',
                        marker='s',
                        zorder=420,
                    )

            ## Ticks
            if i == D - 1:
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45.,
                    horizontalalignment='right',
                    rotation_mode='anchor',
                )
            else:
                ax.xaxis.set_ticks_position('none')

            if i == j:
                ax.set_yticks([])
            elif j == 0:
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
            else:
                ax.yaxis.set_ticks_position('none')

            ## Labels
            if labels is not None:
                if i == D - 1:
                    ax.set_xlabel(labels[j])

                if j == 0 and i != j:
                    ax.set_ylabel(labels[i])

            ax.label_outer()

    figure.align_labels()

    return figure


def pp(
    p: ArrayLike,
    color: Union[str, tuple] = None,
    label: str = None,
    figure: mpl.figure.Figure = None,
    **kwargs,
) -> mpl.figure.Figure:
    r"""P-P plot"""

    # Figure
    if figure is None:
        kwargs.setdefault('figsize', (3.2, 3.2))

        figure, ax = plt.subplots(**kwargs)
        new = True
    else:
        ax = figure.axes.squeeze()
        new = False

    # CDF
    p = np.sort(np.asarray(p))
    p = np.hstack([0, p, 1])
    cdf = np.linspace(0, 1, len(p))

    # Plot
    if new:
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')

    ax.plot(p, cdf, color=color, label=label)

    ax.grid()
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'CDF$(p)$')

    if label is not None:
        ax.legend(loc='upper left')

    return figure
