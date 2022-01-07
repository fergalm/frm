from  ipdb import set_trace as idebug 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

class InteractivePlot(object):
    """An interactive plot
    Clicking in the plot activates a requested function which can
    interact with the data.

    See exampleInteractivePlot() in this file
    """
    def __init__(self, df, xCol, yCol):
        """

        Inputs:
        -------
        data
            (np.ndarray) Data to plot an interact with

        xCol, yCol
            (ints) Which columns of x and y to plot. See note below



        Optional Inputs:
        -----------
        plotFunc
            (function) The plotting function. It's signature must be
            ::
              python  plotFunc(data, xCol, yCol)

        callback
            (function) The function that is called when the plot is
            clicked. It's signature must be
            ::
              callback(data, indexOfRow)
            Index of row is computed by the class to be the row of
            the point closest to where the most was clicked.

            If not specified, a default callback is used that
            just prints out the columns of the requested row.
            see self.defaultCallback()


        Notes:
        ---------
        plotFunc will be passed xCol and yCol, but is not required to
        do anything with them. However, when the plot is clicked,
        the object computes the row that minimises
        ::
          hypot( data[row, xCol] - xClick, data[row, yCol] - yClick)

        and passed that row to the call back function. It will be
        pretty confusing to the user if the xcol and ycol of data
        are not plotted on the screen when this happens.

        """

        #Set input arguments as class variables.
        arg = locals()
        for var in arg:
            setattr(self, var, arg[var])

        self.plotFunc()
        self.fig1 = plt.gcf()
        self.fig2 = plt.figure()
        plt.figure(self.fig1.number)

        self.updateEvent = 'key_press_event'
        self.fig1.canvas.mpl_disconnect(self.updateEvent)
        self.fig1.canvas.mpl_connect(self.updateEvent, self)



    def __del__(self):
        self.disconnect()

    def __call__(self, event):
        print ("call")
        if not event.inaxes:
            return

        x = event.xdata
        y = event.ydata
        key = event.key
        print(x, y, key)

        if key == 'q':
            return 

        plt.figure(self.fig2.number)
        plt.cla()
        self.callback(x, y, key)
        plt.pause(.01)
        plt.figure(self.fig1.number)

    def plotFunc(self,):
        xCol, yCol = self.xCol, self.yCol
        plt.clf()
        plt.plot(self.df[xCol], self.df[yCol], 'C0o')

    def defaultCallback(self, x, y, key):
        row = self.xyToRow(x, y)
        print( key, self.df.iloc[row])

    def xyToRow(self, x, y):
        xvals = self.df[self.xCol]
        yvals = self.df[self.yCol]

        #Work around date issues
        if isinstance(xvals[0], pd.Timestamp):
            print("Is datetime")
            xvals = mdates.date2num(xvals)
        dx = xvals - x
        dy = yvals - y

        #Convert abs distance in each direction to distances relative to axis limits
        #this behaves better when the x and y axes cover dramatically different ranges
        x1, x2, y1, y2 = plt.axis()
        frac_x = dx / (x2 - x1)
        frac_y = dy / (y2 - y1) 

        dist = np.hypot(frac_x, frac_y)
        dist[ ~np.isfinite(dist) ] = 2 * np.max(dist)
        row = np.argmin(dist)
        return row

    def disconnect(self):
        """Disconnect the figure from the interactive object

        This method is called by the destructor, but Python
        doesn't always call the destructor, so you can do so
        explicitly.         And probl should
        """
        print ("Disconnecting...")
        plt.gcf().canvas.mpl_disconnect(self.updateEvent)
