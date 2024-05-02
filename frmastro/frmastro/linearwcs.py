"""
A linear world coordinate system (WCS) object

The Astropy wcs object is a pain to use. This is a 
quick and dirty alternative.

It assumes a linear relationship between pixels and coordinates
that should be good enough for small fields.

How good? Don't publish with it.

Not well tested yet.
"""

from ipdb import set_trace as idebug
import numpy as np

class LinearWcs:
    def __init__(self, crval1, crval2, crpix1, crpix2, cd):
        self.crval = np.eye(3)
        self.crval[2, 0] = crval1 
        self.crval[2, 1] = crval2 

        self.crpix = np.eye(3)
        self.crpix[2, 0] = -crpix1 
        self.crpix[2, 1] = -crpix2 

        self.CD = np.eye(3)
        self.CD[:2, :2] = cd 

        # print(self.crpix)
        # print(self.CD)
        # print("***")
        #I might have the order wrong here
        self.pixToSkyMat = self.crpix @ self.CD @ self.crval
        # print(self.pixToSkyMat) 
        self.skyToPixMat = np.linalg.inv(self.pixToSkyMat)

    def __repr__(self):
        val = [
            "<LinearWcs ",
            f"crpix={self.crpix[2,:2]}",
            f"crval={self.crval[2,:2]}",
            f"CD={self.CD[:2,:2]}",
            ">"
        ]
        return " ".join(val)
    
    def pixToSky(self, col, row): 
        pix = np.hstack([col, row, np.ones_like(col)])
        coords = pix @ self.pixToSkyMat
        #TODO, think about return data types 
        return coords 
    
    def skyToPix(self, ra, dec):
        sky = np.hstack([ra, dec, np.ones_like(ra)])
        pixels = sky @ self.skyToPixMat
        return pixels 
    

    @classmethod
    def from_fitsheader(self, hdr):
        crpix1 = hdr['CRPIX1']
        crpix2 = hdr['CRPIX2']
        crval1 = hdr['CRVAL1']
        crval2 = hdr['CRVAL2']

        if 'CD1_1' in hdr:
            cd = np.zeros((2,2))
            cd[0,0] = hdr['CD1_1']
            cd[0,1] = hdr['CD1_2']
            cd[1,0] = hdr['CD2_1']
            cd[1,1] = hdr['CD2_2']
        else:
            pc = np.zeros((2,2))
            pc[0,0] = hdr['PC1_1']
            pc[0,1] = hdr['PC1_2']
            pc[1,0] = hdr['PC2_1']
            pc[1,1] = hdr['PC2_2']

            cdelt = np.array([
                [hdr['CDELT1'], 0           ], 
                [0,             hdr['CDELT2']],
            ])
            cd = pc @ cdelt

        obj = LinearWcs(crval1, crval2, crpix1, crpix2, cd)
        if 'CD1_1' not in hdr:
            obj.pc = pc 
            obj.cdelt = cdelt 
        return obj





def main():
    import astropy.io.fits as pyfits 
    fn = "/home/fergal/data/jwst/G191-B2B/MAST_2022-10-02T2130/JWST/jw01537011001_04101_00001_mirimage/jw01537011001_04101_00001_mirimage_i2d.fits"

    hdr = pyfits.getheader(fn, 1)

    wcs = LinearWcs.from_fitsheader(hdr)
    # tmp = wcs.pixToSky(518, 511)
    # pix = wcs.skyToPix(tmp[0], tmp[1])
    tmp = wcs.pixToSky(704, 363)
    pix = wcs.skyToPix(tmp[0], tmp[1])

    print(tmp)
    print(pix)
    # print(wcs.pixToSky(518, 511))
    return wcs