# import matplotlib.pyplot as plt 
import numpy as np 



import frmastro.psf.fastgauss as fpf 
from  frmastro.psf. bbox import Bbox


def test_total_flux():
    obj = fpf.FastGaussianModel()

    flux = 20
    sky = 0
    sigma = 2
    params = [20, 20.5, flux, sigma, sky]
    bbox = Bbox(0, 0, 40, 40)
    img = obj.get(bbox, params)

    assert np.isclose(np.sum(img), flux)
    assert np.isclose(img[0,0], sky)

    assert np.isclose(np.max(img), img[20,20])

def test_centroid():
    obj = fpf.FastGaussianModel()

    flux = 20
    sky = 0
    sigma = 2
    bbox = Bbox(0, 0, 40, 40)

    params1 = [20, 20.0, flux, sigma, sky]
    img1 = obj.get(bbox, params1)

    params2 = [20, 20.5, flux, sigma, sky]
    img2 = obj.get(bbox, params2)

    print( img1[19:22, 19:22])
    print( img2[19:22, 19:22])

    #Check flux shifts to higher rows 
    assert np.all(img1[19, :] > img2[19, :])
    assert np.all(img1[20, :] < img2[20, :])
    assert np.all(img1[21, :] < img2[21, :])

