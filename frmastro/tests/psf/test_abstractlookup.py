
from frmastro.psf.bbox import Bbox 
from frmastro.psf.abstractlookup import AbstractLookupPrf
import numpy as np

import matplotlib.pyplot as plt
import frmastro.psf.disp as disp 

class DummyPrf(AbstractLookupPrf):
    def getModelPrfForColRow(self, col:float, row:float):
        pass 


def test_placePrfInBBox_smoke():
    bbox = Bbox(0, 0, 15, 15)

    prfImg = np.ones((3,3))
    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8, 12,0)

    expected = np.zeros(15)
    expected[7:10] = 1 
    assert np.all(result[12,:] == expected)

def test_placePrfInBBox_edges():
    bbox = Bbox(0, 0, 15, 15)
    prfImg = np.ones((3,3))

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 1, 6, 0)
    expected = np.zeros(15)
    expected[:3] = 1 
    assert np.all(result[6,:] == expected)

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 14, 6, 0)
    expected = np.zeros(15)
    expected[13:] = 1 
    assert np.all(result[6,:] == expected)

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8, 0, 0)
    expected = np.zeros(15)
    expected[:2] = 1 
    assert np.all(result[:,8    ] == expected)

    result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8, 14,0)
    expected = np.zeros(15)
    expected[-2:] = 1 
    assert np.all(result[:,8] == expected)


def test_placePrfInBBox_frac():
    bbox = Bbox(0, 0, 15, 15)
    prfImg = np.ones((3,3))

    for frac in np.arange(0,1, .125):
        result = DummyPrf("./").placePrfInBbox(bbox, prfImg, 8+frac, 12+frac, 0)

        expected = np.zeros(15)
        expected[7:10] = 1 
              
        assert np.all(result[12,:] == expected)


