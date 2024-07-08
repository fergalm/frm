from  ipdb import set_trace as idebug 
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


class AbstractInteractivePlot():
    def __init__(self):
        #Set input arguments as class variables.

        self.fig1 = plt.gcf()
        self.fig2 = plt.figure()

        self.updateEvent = 'key_press_event'
        self.fig1.canvas.mpl_disconnect(self.updateEvent)
        self.fig1.canvas.mpl_connect(self.updateEvent, self)

        plt.figure(self.fig1.number)
        plt.clf()
        self.plotFunc()
        plt.pause(.01)


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

    def disconnect(self):
        self.fig1.canvas.mpl_disconnect(self.updateEvent)

    def plotFunc(self,):
        raise NotImplementedError

    def callback(self, x, y, key):
        raise NotImplementedError


class AbstractDataframeInteractivePlot(AbstractInteractivePlot):
    def __init__(self, df, xCol, yCol):
        self.df = df 
        self.xCol = xCol 
        self.yCol = yCol 

        AbstractInteractivePlot.__init__(self) 

    def xyToRow(self, x, y):
        """Utility function to convert x,y to a row in a dataframe"""
        xvals = self.df[self.xCol].values
        yvals = self.df[self.yCol].values

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

class InteractivePlot(AbstractDataframeInteractivePlot):
    """
    
    This is an example implementation

    Clicking in the plot activates the callback function which
    prints the relevant row in the dataframe

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

        AbstractDataframeInteractivePlot.__init__(self, df, xCol, yCol)

    def plotFunc(self,):
        xCol, yCol = self.xCol, self.yCol
        plt.clf()
        plt.plot(self.df[xCol], self.df[yCol], 'C0o')

    def callback(self, x, y, key):
        row = self.xyToRow(x, y)
        print( key, self.df.iloc[row])





class Crosshairs:
    """
    A cross hair cursor.

    Draw a curson consisting of vertical and horizontal lines centered
    on the mouse. The cursor updates with each mouse move.

    """
    def __init__(self, ax=None):
        if ax is None:
            ax = plt.gca()

        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)
    
        plt.gcf().canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


class SnaptoCursor:
    """
    A cursor with crosshair snaps to the nearest x point.
    For simplicity, I'm assuming x is sorted.

    Adapted from https://stackoverflow.com/questions/14338051/matplotlib-cursor-snap-to-plotted-data-with-datetime-axis

    This class needs some work to make it more general. It should ideally work with
    * Normal values and Timestamps
    * numpy arrays and pandas series
    * select x and or y cursor lines


    Usage
    -----------
    ```
    cursor = SnapToCursor(plt.gca(), x, y)
    plt.connect('motion_notify_event', cursor.on_mouse_move)
    ```

    """
    def __init__(self, ax, x, y):
        """
        ax: plot axis
        x: plot spacing. Numpy array of timestamps (series won't work)
        y: plot data
        """
        self.ax = ax
        self.lx = ax.axhline(y = min(y), color = 'k')  #the horiz line
        self.ly = ax.axvline(x = min(x), color = 'k')  #the vert line
        self.x = x
        self.y = y


    def on_mouse_move(self, event):
        if not event.inaxes:
            return

        # import datetime, matplotlib.dates as mdates
        mouseX, mouseY = event.xdata, event.ydata

        #searchsorted: returns an index or indices that suggest where mouseX should be inserted
        #so that the order of the list self.x would be preserved
        #This line assumes self.x is a date. I should generalize
        indx = np.searchsorted(mdates.date2num(self.x), mouseX)
        #if indx is out of bounds
        if indx >= len(self.x):
            indx = len(self.x) - 1

        mouseX = self.x[indx]
        mouseY = self.y[indx]

        self.ly.set_xdata(mouseX)

        # self.txt.set_text(self.format(mouseX, mouseY))
        plt.gca().fmt_xdata = lambda y: pd.to_datetime(self.x[indx]).isoformat()[:19]
        plt.draw()
