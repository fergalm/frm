from ipdb import set_trace as idebug
from pdb import set_trace as debug
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.colors as mcolor
import astropy.io.fits as pyfits
import scipy.ndimage as spimg
import matplotlib as mpl


"""The the filter tracings from the Excel (!) file provided by the
instrument makers

    http://ircamera.as.arizona.edu/MIRI/pces.htm
"""

def main():
    fn = "ImPCE_TN-00072-ATC-Iss2.csv"
    data = np.loadtxt(fn, skiprows=2, delimiter=',')
    data[:,0] *= 1e-6  #Convert to metres

    cols = getColNames(fn)

    for i in range(1, len(cols)):
        print(cols[i])
        trace = np.vstack([data[:,0], data[:,i]]).transpose()
        filename = "miri-%s.dat" %(cols[i])

        np.savetxt(filename, trace, fmt="%.6e")

def getColNames(fn):
    with open(fn) as fp:
        line = fp.readline()[:-1]

    cols = line.split(',')
    return cols

if __name__ == "__main__":
    main()
