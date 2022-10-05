r"""Plotting helpers."""

__all__ = ['nice_rc', 'corner', 'mark_point', 'coverage_plot']

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from numpy import ndarray as Array
from typing import *


def nice_rc(latex: bool = False) -> Dict[str, Any]:
    r"""Returns a dictionary of runtime configuration (rc) settings for nicer
    :mod:`matplotlib` plots. The settings include 12pt font size, higher DPI,
    tight layout, transparent background, etc.

    Arguments:
        latex: Whether to use LaTeX typesetting or not.

    Example:
        >>> plt.rcParams.update(nice_rc())
    """

    rc = {
        'axes.axisbelow': True,
        'axes.linewidth': 0.8,
        'figure.dpi': 150,
        'figure.figsize': (6.4, 4.8),
        'font.size': 12.0,
        'legend.fontsize': 'x-small',
        'lines.linewidth': 1.0,
        'lines.markersize': 3.0,
        'savefig.transparent': True,
        'xtick.labelsize': 'x-small',
        'xtick.major.width': 0.8,
        'ytick.labelsize': 'x-small',
        'ytick.major.width': 0.8,
    }

    if latex:
        rc.update({
            'font.family': ['serif'],
            'font.serif': ['Computer Modern'],
            'text.usetex': True,
        })

    return rc


class LinearAlphaColormap(mpl.colors.LinearSegmentedColormap):
    r"""Linear segmented transparency colormap.

    Arguments:
        color: A color.
        levels: A sequence of levels dividing the domain into segments.
        alpha: The transparency range.
        name: A name for the colormap.
    """

    def __new__(
        self,
        color: Union[str, tuple],
        levels: Array = None,
        alpha: Tuple[float, float] = (0.0, 1.0),
        name: str = None,
    ):
        if name is None:
            if type(color) is str:
                name = f'alpha_{color}'
            else:
                name = f'alpha_{hash(color)}'

        if levels is None:
            levels = [0.0, 1.0]

        levels = np.asarray(levels)
        levels = (levels - levels.min()) / (levels.max() - levels.min())
        alpha = np.linspace(*alpha, len(levels))

        rgb = mpl.colors.to_rgb(color)
        colors = sorted((l, rgb + (a,)) for l, a in zip(levels, alpha))

        return mpl.colors.LinearSegmentedColormap.from_list(
            name=name,
            colors=colors,
        )


def gaussian_blur(img: Array, sigma: float = 1.0) -> Array:
    r"""Applies a Gaussian blur to an image.

    Arguments:
        img: An image array.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The blurred image.

    Example:
        >>> img = np.random.rand(128, 128)
        >>> gaussian_blur(img, sigma=2.0)
        array([...])
    """

    size = 2 * int(3 * sigma) + 1

    k = np.arange(size) - size / 2
    k = np.exp(-(k**2) / (2 * sigma**2))
    k = k / np.sum(k)

    smooth = lambda x: np.convolve(x, k, mode='same')

    for i in range(len(img.shape)):
        img = np.apply_along_axis(smooth, i, img)

    return img


def credible_levels(hist: Array, creds: Array) -> Array:
    r"""Returns the levels of credibility region contours.

    Arguments:
        hist: An histogram.
        creds: The region credibilities.
    """

    x = np.sort(hist, axis=None)[::-1]
    cdf = np.cumsum(x)
    idx = np.searchsorted(cdf, creds * cdf[-1])

    return x[idx]


def corner(
    data: Array,
    weights: Array = None,
    domain: Tuple[Array, Array] = None,
    bins: Union[int, List[int]] = 64,
    creds: Array = [0.6827, 0.9545, 0.9973],
    color: Union[str, tuple] = None,
    alpha: Tuple[float, float] = (0.0, 0.5),
    legend: str = None,
    labels: List[str] = None,
    smooth: float = 0,
    figure: mpl.figure.Figure = None,
    **kwargs,
) -> mpl.figure.Figure:
    r"""Displays each 1 or 2-d projection of multi-dimensional data, as a triangular
    matrix of histograms, known as corner plot. For 2-d histograms, highest density
    credibility regions are delimited.

    Arguments:
        data: Multi-dimensional data, either as a table or as a matrix of histograms.
        weights: The importance weights of the data samples.
        domain: A pair of lower and upper domain bounds. If :py:`None`, inferred from data.
        bins: The number(s) of bins per dimension.
        creds: The region credibilities (in :math:`[0, 1]`) to delimit.
        color: The color of histograms. If :py:`None`, use the color cycler instead.
        alpha: The transparency range.
        legend: A legend.
        labels: The dimension labels.
        smooth: The standard deviation of the smoothing kernels.
        figure: A corner plot over which to draw the new one.
        kwargs: Keyword arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        The figure instance for the corner plot.

    Example:
        >>> data = np.random.randn(2**16, 3)
        >>> labels = [r'$\alpha$', r'$\beta$', r'$\gamma$']
        >>> figure = corner(data, bins=32, labels=labels, figsize=(4.8, 4.8))

    .. image:: ../images/corner.png
        :align: center
        :width: 600
    """

    # Histograms
    if np.asarray(data[0][0]).ndim == 0:
        data = np.asarray(data)
        D = data.shape[-1]

        if type(bins) is int:
            bins = [bins] * D

        if domain is None:
            lower, upper = data.min(axis=0), data.max(axis=0)
        else:
            lower, upper = map(np.asarray, domain)

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
                        weights=weights,
                    )
                else:
                    hist, _, _ = np.histogram2d(
                        data[..., i],
                        data[..., j],
                        bins=(bins[i], bins[j]),
                        density=True,
                        weights=weights,
                    )

                hists[i, j] = hist
    else:
        data = np.asarray(data, dtype=object)
        D = len(data)

        if domain is None:
            lower, upper = np.zeros(D), np.ones(D)
        else:
            lower, upper = map(np.asarray, domain)

        bins = [None] * D
        for i in range(D):
            if data[i, i] is not None:
                bins[i] = np.linspace(lower[i], upper[i], len(data[i, i]) + 1)

        hists = data

    # Figure
    if figure is None:
        kwargs.setdefault('figsize', (6.4, 6.4))

        figure, axes = plt.subplots(
            D,
            D,
            squeeze=False,
            sharex='col',
            gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
            **kwargs,
        )
        new = True
    else:
        axes = np.asarray(figure.axes).reshape(D, D)
        new = False

    # Legend

    ## Quantiles
    creds = np.sort(np.asarray(creds))[::-1]
    creds = np.append(creds, 0)

    if new:
        cmap = LinearAlphaColormap('black', levels=creds, alpha=alpha)

        levels = (creds - creds.min()) / (creds.max() - creds.min())
        levels = (levels[:-1] + levels[1:]) / 2

        for c, l in zip(creds[:-1], levels):
            axes[0, -1].plot(
                [],
                [],
                color=cmap(l),
                linewidth=6,
                solid_capstyle='butt',
                label=r'${:.1f}\,\%$'.format(c * 100),
            )

    ## Color
    lines = axes[0, -1].plot([], [], color=color, solid_capstyle='butt', label=legend)
    color = lines[-1].get_color()

    axes[0, -1].legend(loc='upper right', frameon=False)

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
                hist = gaussian_blur(hist, smooth)

            ## Draw
            x, y = bins[j], bins[i]
            x = (x[1:] + x[:-1]) / 2
            y = (y[1:] + y[:-1]) / 2

            if i == j:
                ax.plot(x, hist, color=color)

                _, top = ax.get_ylim()
                top = max(top, hist.max() * 1.0625)

                ax.set_xlim(left=bins[i][0], right=bins[i][-1])
                ax.set_ylim(bottom=0.0, top=top)
            else:
                levels = np.unique(credible_levels(hist, creds))

                cf = ax.contourf(
                    x,
                    y,
                    hist,
                    levels=levels,
                    cmap=LinearAlphaColormap(color, levels, alpha=alpha),
                )
                ax.contour(cf, colors=color)

                if j > 0:
                    ax.sharey(axes[i, j - 1])
                else:
                    ax.set_ylim(bottom=bins[i][0], top=bins[i][-1])

            ## Ticks
            if i == D - 1:
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune='both'))
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45.0,
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
    figure.tight_layout(pad=0.5)

    return figure


def mark_point(
    figure: mpl.figure.Figure,
    point: Array,
    color: Union[str, tuple] = 'black',
    linestyle: str = 'dashed',
    marker: str = 's',  # square
    legend: str = None,
) -> None:
    r"""Marks a point on the histograms of a corner plot.

    Arguments:
        figure: The corner plot figure (see :func:`corner`).
        point: The point to mark.
        color: The color of the lines and markers.
        linestyle: The style of the lines.
        marker: The shape of the markers.
        legend: A legend.

    Example:
        >>> mark_point(figure, [0.5, 0.3, -0.7], color='black')

    .. image:: ../images/corner_marked.png
        :align: center
        :width: 600
    """

    D = len(point)
    axes = np.asarray(figure.axes).reshape(D, D)

    # Legend
    if legend is not None:
        axes[0, -1].plot(
            [],
            [],
            color=color,
            linestyle=linestyle,
            marker=marker,
            solid_capstyle='butt',
            label=legend,
        )

        axes[0, -1].legend(loc='upper right', frameon=False)

    # Plot
    for i in range(D):
        for j in range(i + 1):
            ax = axes[i, j]
            ax.axvline(point[j], color=color, linestyle=linestyle)

            if i != j:
                ax.axhline(point[i], color=color, linestyle=linestyle)
                ax.plot(point[j], point[i], color=color, marker=marker)


def coverage_plot(
    levels: Array,
    coverages: Array,
    color: Union[str, tuple] = None,
    legend: str = None,
    figure: mpl.figure.Figure = None,
    **kwargs,
) -> mpl.figure.Figure:
    r"""Plots the expected coverage at various credible levels.

    Arguments:
        levels: A vector of increasing credible levels.
        coverages: A vector of corresponding expected coverages.
        legend: A legend.
        figure: Another coverage plot over which to draw the new one.
        kwargs: Keyword arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        The figure instance for the coverage plot.

    Example:
        >>> levels = np.linspace(0, 1, 512) ** 2
        >>> coverages = np.linspace(0, 1, 512)
        >>> figure = coverage_plot(levels, coverages)

    .. image:: ../images/coverage.png
        :align: center
        :width: 400
    """

    levels, coverages = np.asarray(levels), np.asarray(coverages)

    # Figure
    if figure is None:
        kwargs.setdefault('figsize', (3.2, 3.2))

        figure, ax = plt.subplots(**kwargs)
        new = True
    else:
        ax = figure.axes[0]
        new = False

    # Plot
    if new:
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')

    ax.plot(levels, coverages, color=color, label=legend)

    ax.grid()
    ax.set_xlabel(r'Credible level')
    ax.set_ylabel(r'Expected coverage')

    if legend is not None:
        ax.legend(loc='upper left')

    figure.tight_layout(pad=0.5)

    return figure
