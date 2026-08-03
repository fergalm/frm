# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:16:11 2026

@author: e085120
"""

from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
import numpy as np
import warnings

import frmplots.plots as fplots
import matplotlib.colors as mcolor


class RainbowPlotter:
    """
    Chart the evolution of a timeseries by plotting each subsequent
    iteration in a slightly different colour. For example, suppose
    `y = f(x | alpha)`

    We might want to plot x against y for a variety of different values
    of alpha, using colour to distinguish the different values of alpha.

    Example
    -------------
    ```
    rp = RainbowPlotter(10)
    x = np.linspace(0, 1, 100)
    for alpha in range(10):
        y = f(x, alpha)
        rp.plot(x, y)
    ```

    The alpha=0 curve will be plotted in red, the alpha=9 curve in purple,
    and other values of alpha in all the colours of the rainbow.
    """

    def __init__(self, num, cmap=plt.cm.rainbow_r, norm=None):
        self.num = num
        self.cmap = cmap

        if norm is None:
            if num > 0:
                norm = mcolor.BoundaryNorm(np.arange(self.num), cmap.N)
        self.norm = norm
        self.count = 0

        self.promise = []

    def plot(self, x, y, *args, **kwargs):
        """
        Plot x against y using `plt.plot`. To use
        a different plotting function, set plotFunc as a keyword
        argument

        Inputs
      res   ------------
        x, y  (Sequences)
            Values to plot

        Optional Arguments
        ---------------------
        plotFunc
            Function to actually do the plotting. Default is `plt.plot `

        All other optimal arguments are passed to `plotFunc`
        Example
        -----------
        ```
            plot(x, y, plotFunc=plt.step)
        ```
        """

        plotFunc = kwargs.pop("plotFunc", plt.plot)

        clr = self.cmap(self.count / self.num)
        self.count += 1

        handle = plotFunc(x, y, *args, color=clr, **kwargs)
        return handle

    def plot_groups(self, df, key, xCol, yCol, *args, **kwargs):
        """Conveience function to plot every group in a dataframe

        Not as flexible as calling plot directly, but handles the
        most common usecase.
        """
        gr = df.groupby(key)
        self.num = len(gr)
        self.norm = mcolor.BoundaryNorm(np.arange(self.num), self.cmap.N)

        for key in gr.groups.keys():
            df2 = gr.get_group(key)
            self.plot(df2[xCol], df2[yCol], *args, **kwargs)

    #Sketch of an implementation where you don't know the
    #number of plots in advance
    # def future(self, x, y, *args, **kwargs):
    #     package = [x, y, args, kwargs]
    #     self.promise.append(package)

    # def render(self):
    #     self.num = len(self.promise)
    #     for prom in self.promise:
    #         x, y, args, kwargs = prom
    #         self.plot(x, y, *args, **kwargs)


    def colorbar(self, label=None):
        cb = fplots.create_colorbar(self.cmap, self.norm)
        if label:
            cb.set_label(label)
        return cb

    def add_colorbar(self, label=None):
        warnings.warn("Deprecated. Use colorbar() instead", DeprecationWarning)
        return self.colorbar(label)