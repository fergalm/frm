from ipdb import set_trace as idebug 
import matplotlib.pyplot as plt 
from pprint import pprint 
import pandas as pd 
import numpy as np 

import frmbase.support as fsupport 
import frmbase.meta as fmeta 
lmap = fsupport.lmap 

from frmpolitics.census import CensusQuery, TigerQueryAcs
from frmgis.geomcollect import GeomCollection
import frmgis.get_geom as fgg 

import frmgis.plots as fgplots 
import getincome as gi 


"""Checking my overlap calculation make sense
"""

def main():
    alice = pd.read_csv('tmp-alice.csv')
    pattern = '/home/fergal/data/elections/shapefiles/schools/{school_type}_School_Districts.kml'
    sch = gi.load_all_schools(pattern)

    sch_geom = sch[sch.Name == 'Dumbarton MS'].geom

    plt.clf()
    fgplots.plot_shape(sch_geom, 'k-')