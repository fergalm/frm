# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:12:23 2020

@author: fergal
"""

from pdb import set_trace as debug
import matplotlib.pyplot as plt

from frmgis.anygeom import AnyGeom
import frmplots.plots as fplots


def test():
    plt.plot([0,10], [0,10], 'k.')
    x = AoiMarker()
    return x

class AoiMarker():
    """Interactively draw a geometry.

    For most work you want to draw geometries and AOIs in Google
    Earth. Sometimes though, it's quicker to draw them from
    Python. This class enables interactive drawing of polygons
    using matplotlib.

    The current implementation is restricted to drawing simple polygons,
    but expanding the feature set to include any other geometries
    would not be too much work.

    Usage
    ------
    ::

        #Create a plot
        ...

        obj = AoiMarker()
        #Click on points on the plot to create a polygon...

        wkt = obj.get_anygeom().as_wkt()


    Todo
    -----
    Animate drawing of polygons so a line always follows the
    cursor.
    """

    def __init__(self):
        """

        Inputs:
        -------
        data
            (np.ndarray) Data to plot an interact with

        xCol, yCol
            (ints) Which columns of x and y to plot. See note below

        plotFunc
            (function) The plotting function. It's signature must be
            ::
              python  plotFunc(data, xCol, yCol)


        Optional Inputs:
        -----------
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

        f = plt.gcf()
        # self.updateEvent = 'button_release_event'
        self.updateEvent = 'key_press_event'
        f.canvas.mpl_disconnect(self.updateEvent)
        f.canvas.mpl_connect(self.updateEvent, self)
        self.point_list = []
        self.line_obj = None

        self.is_complete = False

    def __del__(self):
        self.disconnect()

    def __call__(self, event):
        print ("call")
        if not event.inaxes:
            print ("Event not in axis")
            return

        print(event)
        if event.key == 'm':
            x = event.xdata
            y = event.ydata
            self.point_list.append([x,y])
            self.draw_geom()
        elif event.key == 'q':
            self.is_complete = True

    def draw_geom(self):

        if self.line_obj is not None:
            pass
            #Remove old line
            # self.line_obj.pop(0).remove()

        tmp = AnyGeom(self.point_list, 'POLYGON')
        print(tmp.as_wkt())
        self.line_obj = fplots.plot_shape(tmp, 'r-')
        plt.title(len(self.line_obj))
        plt.pause(.001)

    def get_anygeom(self):
        if len(self.point_list) == 0:
            shape = AnyGeom("POINT EMPTY")
        else:
            points = self.point_list + [self.point_list[0]]
            shape = AnyGeom(points, 'POLYGON')

        return shape

    def disconnect(self):
        """Disconnect the figure from the interactive object

        This method is called by the destructor, but Python
        doesn't always call the destructor, so you can do so
        explicitly.         And probl should
        """
        print ("Disconnecting...")
        plt.gcf().canvas.mpl_disconnect(self.updateEvent)
