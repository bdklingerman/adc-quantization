'''
Visualize.py serves as the internal tool to turn numpy and pandas data
into visual plots.

Author: Brad Klingerman (bdklingerman)
Version: September 18, 2026
'''

import matplotlib.pyplot
import numpy


def save_plot(
        y: numpy.ndarray,
        x: numpy.ndarray | None = None,
        *,
        xlabel: str = "",
        ylabel: str = "",
        filename: str | None = None
) -> tuple[matplotlib.pyplot.Figure, matplotlib.pyplot.Axes]:
    '''Save a new figure to storage.'''

    fig, ax = matplotlib.pyplot.subplots()
    if x is not None:
        ax.plot(x,y)
        ax.set_xlim(x[0],x[-1])
    else:
        ax.plot(y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if filename is not None:
        fig.savefig(f"data/{filename}.png", dpi=600)

    return fig, ax