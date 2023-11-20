# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:47:30 2023

@author: fergal
"""

from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import frmplots.platlabels as pl
import frmgis.mapoverlay as fmo

def test_smoke():
    plt.clf()
    plt.axis([0, 10, 0, 10])

    obj = pl.PlatLabel()

    for i in range(10):
        obj.text(3 + i/4., 5, f"Text {i}")
            # obj.drawBbox(obj.tList[i])
    obj.render()

    for i in range(10):
        if i in [0, 4, 8]:
            assert obj.tList[i].get_visible()
        else:
            assert ~obj.tList[i].get_visible()

def test_advanced_render():
    plt.clf()
    plt.axis([0, 10, 0, 10])

    obj = pl.AdvancedPlatLabel()

    for i in range(10):
        x = 3 + i/2 
        y = 5 - (i%3)/10
        plt.plot([x], [y], 'o')
        obj.text(x, y, f"Text {i}", fontsize=10)
            # obj.drawBbox(obj.tList[i])
    obj.render()

    for i in range(10):
        if i in [0, 1, 3, 4, 6, 7, 9]:
            assert obj.tList[i].get_visible()
        else:
                assert ~obj.tList[i].get_visible()



def test_plat():
    """Done on a plane with no internet. this is horrible. Delete it.
    I shouldn't be using a background image at all, I don't think,
    definitely not one written to disk"""

    fn = "tmp.png"
    extent = [-76.92599135, -76.25729365, 39.12838905, 39.74953595]

    plt.clf()
    img = plt.imread(fn)
    plt.imshow(img, extent=extent)


    obj = pl.PlatLabel()
    for i in range(10):
        obj.text(-76.8 + i/40, 39.5, f"Text {i}", level=10-i, fontsize=10+i)
    obj.render()

    for i in range(10):
        if i in [2,5,9]:
            assert obj.tList[i].get_visible()
        else:
                assert ~obj.tList[i].get_visible()
