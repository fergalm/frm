from frmastro.psf.keplerprf import KeplerPrf, K2Prf
from frmastro.psf.bbox import Bbox 
import numpy as np 

"""
These are interface tests. They don't test
that implementation, i.e the the produced
PRFs are correct. I should write some of those
"""

class DummyKeplerPrf(KeplerPrf):
    def readPrfFile(self, position):
        print("Mocking file IO for unit test")

        img = np.ones((1200, 1200))
        return img 


def test_smoke():

    bbox = Bbox.fromCCRR(200, 220, 300, 320)
    path = "/home/fergal/data/keplerprf/"
    obj = DummyKeplerPrf(path, 2, 1)

    params = [210, 310, 1, 0]
    img = obj.get(bbox, params)

    assert np.all(img.shape == bbox.shape)


