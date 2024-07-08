
from frmastro.psf.bbox import Bbox 
from frmastro.psf.abstractlookup import AbstractLookupPrf
import numpy as np

import matplotlib.pyplot as plt
import frmastro.psf.disp as disp 

class DummyPrf(AbstractLookupPrf):
    def getInterpRegPrfForColRow(self, col:float, row:float):
        pass 


def test_placePrfInBBox_smoke():
    bbox = Bbox(0, 0, 15, 15)

    prfImg = np.ones((3,3))
    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8, 12)

    expected = np.zeros(15)
    expected[7:10] = 1 
    assert np.all(result[12,:] == expected)

def test_placePrfInBBox_edges():
    bbox = Bbox(0, 0, 15, 15)
    prfImg = np.ones((3,3))

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 1, 6)
    expected = np.zeros(15)
    expected[:3] = 1 
    assert np.all(result[6,:] == expected)

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 14, 6)
    expected = np.zeros(15)
    expected[13:] = 1 
    assert np.all(result[6,:] == expected)

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8, 0)
    expected = np.zeros(15)
    expected[:2] = 1 
    assert np.all(result[:,8    ] == expected)

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8, 14)
    expected = np.zeros(15)
    expected[-2:] = 1 
    assert np.all(result[:,8] == expected)


def test_placePrfInBBox_frac():
    bbox = Bbox(0, 0, 15, 15)
    prfImg = np.ones((3,3))

    for frac in np.linspace(0,1, 8):
        result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8+frac, 12+frac)

        expected = np.zeros(15)
        expected[7:10] = 1 
        assert np.all(result[12,:] == expected)



# def test_children():
#     """Yes, this is a horrible idea"""

#     from frmastro.psf.keplerprf import KeplerPrf, K2Prf
#     from frmastro.psf.tessfitsprf import TessFitsPrf
#     from frmastro.psf.webbpsf import MiriPsf
#     bbox = Bbox.fromCCRR(200, 220, 300, 320)

#     path = "/home/fergal/data/keplerprf/"
#     kep = KeplerPrf(path, 2, 1)
#     k2 = K2Prf(path, 2, 1)
#     tess = TessFitsPrf(path, 1, 1, 1)
#     webb = MiriPsf("/home/fergal/data/jwst/webbpsf-data/MIRI/psf/PSF_MIRI_in_flight_opd_filter_F1500W.fits")

#     params = [210, 310, 1, 0]
#     img = kep.get(bbox, params)
#     img = k2.get(bbox, params)
#     img = tess.get(bbox, params)
#     img = webb.get(bbox, params)

#     plt.clf()
#     disp.plotImage(img)