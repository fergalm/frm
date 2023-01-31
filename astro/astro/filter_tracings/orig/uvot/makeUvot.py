#!/usr/bin/python

import matplotlib.pyplot as mpl
import numpy as np
import pyfits

import const
def generateFilter(fitsFilename, outFilename, filterName=""):
    data = pyfits.getdata(fitsFilename)

    nrgLow = data['ENERG_LO']
    nrgHigh = data['ENERG_HI']
    transmission = data['MATRIX']

    nrg_kev = (nrgLow +nrgHigh)/2.
    nrg_ev = nrg_kev * 1000
    nrg_J = nrg_ev * const.electronCharge

    wavel = const.planck * const.speedOfLight / nrg_J

    #Transmission is given in "effective area". To get fractional transmission
    #divide by actual observing area. Taken from
    #http://heasarc.nasa.gov/docs/swift/about_swift/Sci_Fact_Sheet.pdf
    #at http://heasarc.nasa.gov/docs/swift/about_swift/
    totalArea = 659
    transmission /= totalArea

    #Test that each element of matrix is a float, and not a vector.
    #I don't know what to do with the vectors
    for t in transmission:
        assert( type(t) == np.float32)

    text = []
    text.append('#Swift UVOT %s. From makeUvot.py' %(filterName))
    text.append('#http://heasarc.nasa.gov/docs/swift/proposals/swift_responses.html')

    for w,t in zip(wavel, transmission):
        text.append( "%.3e %.3e" %(w, t))

    fp = open(outFilename, "w")
    fp.write( "\n".join(text))
    fp.close()
    print "Written to %s" %(outFilename)



def main():
    generateFilter('u.rsp', 'uvot_u.dat', filterName='U band')
    generateFilter('b.rsp', 'uvot_b.dat', filterName='B band')
    generateFilter('v.rsp', 'uvot_v.dat', filterName='V band')
    generateFilter('uvw1.rsp', 'uvot_w1.dat', filterName='W1 band')
    generateFilter('uvm2.rsp', 'uvot_m2.dat', filterName='M2 band')
    generateFilter('uvw2.rsp', 'uvot_w2.dat', filterName='W2 band')
    generateFilter('white.rsp', 'uvot_white.dat', filterName='white light')

