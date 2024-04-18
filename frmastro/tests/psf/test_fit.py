import matplotlib.pyplot as plt 
import numpy as np 

from  frmastro.psf. bbox import Bbox
import frmastro.psf.fastgauss as fpf
import frmastro.psf.airy as fpa  
import frmastro.psf.fit as fit 


np.random.seed(12345)

def test_gauss():
    obj = fpf.FastGaussianModel()

    flux = 2000
    sky = 0
    sigma = 2
    params = [20, 20.5, flux, sigma, sky]
    bbox = Bbox(0, 0, 40, 40)
    img = obj.get(bbox, params)
    img += .1 * img * np.random.randn(*img.shape)  #Add some noise
    # img += np.random.randn(*img.shape)  #Add some sky

    params = [20, 19.0, 1, sigma, sky]
    result = fit.fit(img, obj, params, bbox, bounds=None)

    # assert result.success is True 
    diff = img - obj.get(bbox, result.x)
    assert np.all(np.fabs(diff) < 5)
    
    # plt.clf()
    # import frmastro.psf.disp as disp 
    # return disp.threeplot(img, obj, result.x)
    

def test_airy():
    obj = fpa.AiryPrf()

    flux = 1.
    sky = 0.
    sigma = .5
    params = [20, 20.5, flux, sigma, sky]
    bbox = Bbox(0, 0, 40, 40)
    img = obj.get(bbox, params)
    img += .1 * img * np.random.randn(*img.shape)
    # img += np.random.randn(*img.shape)  #Add some sky

    params = [20., 20.0, flux, sigma, sky]
    result = fit.fit(img, obj, params, bbox)
    print(result)
    plt.clf()
    import frmastro.psf.disp as disp 
    print("This test is failing")
    return disp.threeplot(img, obj, result.x)
    

def test_smoke():
    assert False 