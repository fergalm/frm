
from  ipdb import set_trace as idebug 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

import frm.plots as fplots


class PlotWheel(object):
    """An interactive plot
    Clicking in the plot activates a requested function which can
    interact with the data.

    This is a base class extend it with a class that implements a 
    plotting function
    """
    def __init__(self, index_max, i0=0):
        #Set input arguments as class variables.
        self.index = i0
        self.index_max = index_max
        
        f = plt.gcf()
        self.draw_plot(self.index)

        self.updateEvent = 'key_press_event'
        f.canvas.mpl_disconnect(self.updateEvent)
        f.canvas.mpl_connect(self.updateEvent, self)


    def __del__(self):
        self.disconnect()

    def __call__(self, event):
        request = event.key

        if request == 'n':
            if self.index == self.index_max - 1:
                print("Already at end")
                return 
            self.index += 1
        elif request == 'b':
            if self.index == 0:
                print("Already at start")
                return 
            self.index -= 1
        elif request == 'S':
            fn = self.get_savefile_name(self.index)
            plt.savefig(fn)
            print("Saved to %s" %(fn))
        elif request == 'q':
            return

        self.draw_plot(self.index) 

    def get_savefile_name(self, i):
        return "frame%04i.png" %(i)

    def disconnect(self):
        """Disconnect the figure from the interactive object

        This method is called by the destructor, but Python
        doesn't always call the destructor, so you can do so
        explicitly.         And probl should
        """
        print ("Disconnecting...")
        plt.gcf().canvas.mpl_disconnect(self.updateEvent)

    def draw_plot(self, i):
        """Over ride this method to do setup/tear down for each plot. 

        See GroupPlotWheel for an example
        """
        return self.plotFunc(i)

    def plotFunc(self, i):
        raise NotImplementedError


class GroupPlotWheel(PlotWheel):
    """Base class for plotting the results of a groupby"""
    def __init__(self, df, col, i0=0):
        self.gr = df.groupby(col)
        self.keys = list(self.gr.groups.keys())

        PlotWheel.__init__(self, len(self.keys), i0=i0)

    def draw_plot(self, i):
        name = self.keys[i]
        df = self.gr.get_group(name) 
        
        plt.clf()
        self.plotFunc(df, name, i)
        plt.pause(.01)

    def get_savefile_name(self, i):
        name = self.keys[i]
        name = re.sub(" ", "_", name)
        return "%s.png" %(name)

    def plotFunc(sef, df, name, i):
        """

        Implement this method in the child class to create the particular plot you want.

        If you need information for data outside of a single group to create your plot,
        reimpement draw_plot() instead.

        Inputs
        ---------        
        df
            (Pandas Dataframe) Data from this group
        name
            (string?) Unique identifier, or key, for this group
        i
            (int) A number between 0 and the number of groups that can be plotted 
        """
        raise NotImplementedError
