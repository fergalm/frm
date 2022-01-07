#Copyright 2017-2018 Orbital Insight Inc., all rights reserved.
#Contains confidential and trade secret information.
#Government Users: Commercial Computer Software - Use governed by
#terms of Orbital Insight commercial license agreement.

"""
Created on Mon Feb 26 16:54:54 2018

@author: fergal


"""
from __future__ import print_function
from __future__ import division

from pdb import set_trace as debug
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import sphharm

def createSampleData():
    """Create some sample data to fit with \ell values of 0,1 and 2"""
    coeff = [-.76, 0, -.16, .29, -.10, .10, .43, -.56, .50]

    #Number of data points to fit
    num = 1000
    lng_deg = 360 * np.random.rand(num)
    lat_deg = 180 * (np.random.rand(num) - .5)

    computeYlm = sphharm.computeYlm
    data  = coeff[0] * computeYlm(0,0, lng_deg, lat_deg)
    data += coeff[1] * computeYlm(1,-1, lng_deg, lat_deg)
    data += coeff[2] * computeYlm(1,0, lng_deg, lat_deg)
    data += coeff[3] * computeYlm(1,+1, lng_deg, lat_deg)
    data += coeff[4] * computeYlm(2,-2, lng_deg, lat_deg)
    data += coeff[5] * computeYlm(2,-1, lng_deg, lat_deg)
    data += coeff[6] * computeYlm(2,0, lng_deg, lat_deg)
    data += coeff[7] * computeYlm(2,1, lng_deg, lat_deg)
    data += coeff[8] * computeYlm(2,2, lng_deg, lat_deg)

    if False:
        plt.clf()
        plt.scatter(lng_deg, lat_deg, c=data, s=80)
        plt.colorbar()

    return lng_deg, lat_deg, data


def test1():

    lng_deg, lat_deg, data = createSampleData()
    fObj = sphharm.SphHarm(lng_deg, lat_deg, data, None, 2)

    resid = fObj.getResiduals()
    assert np.sum(resid < 1e-10)
