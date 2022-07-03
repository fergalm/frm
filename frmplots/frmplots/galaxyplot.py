
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np


def galaxyPlot(df, colList, *args, **kwargs):
    """Plot everything against everything

    Just like extra galactic astronomers are wont to do.

    Inputs:
    -------------
    df
        (Dataframe) Data to be plotted

    colList
        (list) list of column numbers to plot. Must have at least
        two elements

    Optional Inputs:
    ---------------
    plotFunc
        (A plotting function) What function should actually plot the x,y
        points. Default is plt.plot()

    labels
        (list of strings) Applied as x and y axis labels. If not
        given, defaults to values in colList

    wantLog
        (list or array) If True, given column is plotted in log scale.
        len(wantLog) == len(colList)

    limits
        (list) If the ith elememt of this array is an object of length 2,
        use those two number to set the plot limits in that dimension.
        See notes, below.

    All other optional arguments are passed to plt.plot (or whatever plotFunc
    is)

    Returns:
    -------
    void

    Output:
    --------
    A plot is produced.


    Notes:
    -------
    The signature of plotFunc is
    ``plotFunc(x, y, *args, **kwargs)``
    similar to plt.plot()'s most commonly called style.

    The easiest way to create the limits argument is as follows::

      limits = list(np.zeros_like(colList))
      limits[4] = [1, 10]

    The function will see that only the 4th element of the array
    is a set of limits, and default values will be chosen for all
    other dimensions

    TODO
    ------
    * Adapt code to convert input numpy arrays to dataframes.
    * colList should be an optional argument

    """

    labels = kwargs.pop('labels', colList)
    ms = kwargs.pop('markersize', 3)
    plotFunc = kwargs.pop('plotFunc', plt.plot)

    wantLog = np.zeros(len(colList))
    wantLog = kwargs.pop('wantLog', wantLog)
    assert( len(wantLog) == len(colList))

    limits = np.zeros(len(colList), dtype=bool)
    limits = kwargs.pop('limits', limits)
    assert( len(limits) == len(colList))

    #TODO Convert numpy to dataframe if numpy is input

    nCol = len(colList)
    nPlot = nCol-1

    #Create a 2d array of Axes objects
    axArray = np.empty( (nCol, nCol), dtype=object)
    for i in range(nCol):
        for j in range(0, i):
            n = (nPlot)*(i-1)+j + 1
            sharex, sharey = findSharedAxes(axArray, i, j)

            if sharex is None and sharey is None:
                ax = plt.subplot(nPlot, nPlot, n)
            if sharex is None and sharey is not None:
                ax = plt.subplot(nPlot, nPlot, n, sharey=sharey)
            if sharex is not None and sharey is None:
                ax = plt.subplot(nPlot, nPlot, n, sharex=sharex)
            if sharex is not None and sharey is not None:
                ax = plt.subplot(nPlot, nPlot, n, sharex=sharex, sharey=sharey)

            axArray[i,j] = ax

    #i loops over the y-axes (the rows)
    for i in range(nCol):

        #Pick y limits for subplot. If not supplied, use range of data
        if hasattr(limits[i], "__len__") and len(limits[i]) == 2:
            yLwr, yUpr = limits[i]
        else:
            yLwr = np.nanmin(df[colList[i]])
            yUpr = np.nanmax(df[colList[i]])

        #j loops over the x-axes (the columns)
        for j in range(0, i):
            ax = axArray[i,j]
            plt.sca(ax)

            plotFunc(df[colList[j]], df[colList[i]], \
                markersize=ms, *args, **kwargs)

            #Add decorations for the subplot
            if j== i-1:
                plt.title(labels[j])

            if j == 0:
                plt.ylabel(labels[i])

            if i != nCol-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel(labels[j])

            if wantLog[i]:
                ax.set_yscale('log')
                ax.yaxis.set_tick_params(which="major", length=10)
                ax.yaxis.set_tick_params(which="minor", length=4)


            if wantLog[j]:
                ax.set_xscale('log')
                ax.xaxis.set_tick_params(which="major", length=10)
                ax.xaxis.set_tick_params(which="minor", length=4)

            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

            #Pick x limits for subplot. If not supplied, use range
            #range of data
            if hasattr(limits[j], "__len__") and len(limits[j]) == 2:
                plt.xlim(limits[j])
            else:
                xLwr = np.min(df[colList[j]])
                xUpr = np.max(df[colList[j]])
                plt.xlim([xLwr, xUpr])
            plt.ylim([yLwr, yUpr])

    plt.subplots_adjust(wspace=0, hspace=0)


def findSharedAxes(axArray, i, j):
    xAx = None
    yAx = None

    if i>j+1:
        xAx = axArray[i-1, j]

    if j>0:
        yAx = axArray[i, j-1]

    return xAx, yAx
