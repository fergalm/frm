from frmastro.psf.tessfitsprf import TessFitsPrf
from frmastro.psf.bbox import Bbox 
import numpy as np 

"""
These are interface tests. They don't test
that implementation, i.e the the produced
PRFs are correct. I should write some of those
"""

class DummyTessPrf(TessFitsPrf):
    def loadSingleImage(self, position):
        print(f"Mocking file IO for {position}")

        img = np.ones((1206, 1206))
        return img 


def test_smoke():

    bbox = Bbox.fromCCRR(200, 220, 300, 320)
    path = "/home/fergal/data/keplerprf/"
    obj = DummyTessPrf(path, 2, 1, 1)

    params = [210, 310, 1, 0]
    img = obj.get(bbox, params)

    assert np.all(img.shape == bbox.shape)


