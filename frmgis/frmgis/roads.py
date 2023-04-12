from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt 
from pprint import pprint 
import pandas as pd 
import numpy as np 
import os 

from ipdb import set_trace as idebug 

import frmbase.dfpipeline as dfp 
import frmplots.plots as fplots
import frmgis.plots as fgplots 
import frmgis.get_geom as fgg 


"""
Plot the interstate system in Baltimore county.

Intended as a mapoverlay on chloropleth plots

This isn't very sophisiticated. There is no way to specify which
roads you want, or exactly where to put the annotations
"""

def plot_interstate(clr='#FEEFC3', edgecolor='#F3C04F', annotate=True):
    path = os.path.dirname(__file__)

    df = load(os.path.join(path, 'data/MarylandInterstates.kml'))
    plot_roads(df, clr, edgecolor)
    if annotate:
        annotate_interstate(path)
              

class LoadGeom(dfp.AbstractStep):
    def __init__(self, fn):
        self.fn = fn 

    def apply(self, df=None):
        return fgg.load_geoms_as_df(self.fn)


def load(fn):

    cols = ['geom', 'COUNTY', 'ROADNAMESHA', 'ID_PREFIX', 'ID_RTE_NO',
        'SHAPESTLength']

    pipeline = [
        LoadGeom(fn),
        dfp.Filter('ID_PREFIX =="IS"'),
        dfp.Filter('SHAPESTLength > 0'),
        dfp.SelectCol(cols),
    ]
    df = dfp.runPipeline(pipeline)
    return df 


def plot_roads(df, clr, edgecolor):
    effect = fplots.outline(clr=edgecolor, lw=5)
    for g in df.geom:
        fgplots.plot_shape(g, '-', lw=3, color=clr, path_effects=effect)




def annotate_interstate(path):
    # path = os.path.dirname(__file__)
    # path = "/home/fergal/data/elections/shapefiles/mdroads/I-{num}.png"
    path = os.path.join(path, "data/I-{num}.png")
    num = 70
    coords = np.array([
        [-76.803, 39.298]
    ])
    plot_road_markers(coords[:,0], coords[:,1], num)

    num = 83
    coords = np.array([
        [-76.665, 39.537],
        [-76.647, 39.337],
    ])
    plot_road_markers(coords[:,0], coords[:,1], num)

    num = 95
    coords = np.array([
        [-76.380, 39.438],
        [-76.732, 39.214],
    ])
    plot_road_markers(coords[:,0], coords[:,1], num)

    num = 695
    coords = np.array([
        [-76.557, 39.400]
    ])
    plot_road_markers(coords[:,0], coords[:,1], num)

    num = 795
    coords = np.array([
        [-76.826, 39.447]
    ])
    plot_road_markers(coords[:,0], coords[:,1], num)


def plot_road_markers(x, y, rt_number, ax=None):
    """
    stolen from 
    https://stackoverflow.com/questions/2318288/how-to-use-custom-png-image-marker-with-plot
    """
    rt_number = str(rt_number)
    #TODO Make more general!
    path = f"/home/fergal/data/elections/shapefiles/mdroads/I-{rt_number}.png"
    image = plt.imread(path)

    ax = ax or plt.gca()
    for xi, yi in zip(x,y):
        im = OffsetImage(image, zoom=18/ax.figure.dpi)
        im.image.axes = ax
        ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)
        ax.add_artist(ab)


