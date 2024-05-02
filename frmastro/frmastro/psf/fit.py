from ipdb import set_trace as idebug
from typing import Sequence, List, Tuple
import scipy.optimize as spOpt
import numpy as np


from .abstractprf import AbstractPrfModel
from .bbox import Bbox


def fit(img: np.ndarray, model:AbstractPrfModel, initGuess: Sequence, bbox:Bbox=None, bounds: List[Tuple]=None, mask:np.ndarray=None) -> spOpt.OptimizeResult:
    """
    Fit a PRF model to an image
    
    Uses the L-BFGS-B optimizer in `scipy.optimize`
    
    
    Inputs
    ------------
    - img 
        - A 2d numpy array representing an image 
    - model 
        - An object representing the model the be fit. See `AbstractPrfModel`.
          When in doubt, the `fastgauss.FastGaussianModel` is a good place to start
    - initGuess 
        - Initial guess at the best fit parameters of the model. The contents
          of this list depends on the `model` being used. 
    
    Optional Inputs 
    ------------------
    - bbox
        - If supplied, trim the image to just this bbox before fitting.
          The parameters supplied in `initGuess` should reference this trimmed
          image, as will be the return values
    - bounds 
        Min and max values allowed for the best fit parameters. If a list of 
        2-tuples. bounds[i] = ( min_allowed_value_for_param_i, max_allowed_value_param_i). See `scipy.optimize` for more details 
    - mask 
        - A 2d boolean numpy array with the same shape as `img`. If mask[i,j] 
          is **False**, then the corresponding pixel in `img` is ignored for 
          for the fit
    
    Returns 
    ------------
    A `scipy.optimize.OptimizeResult` object. The best fit parameters are in the
    the `result.x` property
    """
    
    #Set some default values for optional args 
    bbox = bbox or Bbox.fromImage(img)
    bounds = None or model.getDefaultBounds(bbox)           

    if mask is None:
        mask = np.ones_like(img, dtype=bool)

    assert np.all(mask.shape == img.shape)


    if len(initGuess) != len(bounds):
        raise ValueError(f"Parameter initGuess has {len(initGuess)} elements, but bounds has {len(bounds)}")
    
    #Call the model once to catch any errors before we start 
    model.get(bbox, initGuess)
    
    callback = Callback(model, img, bbox, mask)
    
    soln = spOpt.minimize(
        costFunc, 
        initGuess, 
        args=(model, img, bbox, mask), 
        method="L-BFGS-B", 
        bounds=bounds,
        # callback=callback,
        # options = {'disp': True}
    )
    return soln



def costFunc(params:Sequence, model: AbstractPrfModel, img:np.ndarray, bbox, mask) -> float:
    """Compute a metric of difference between image and its model for given model params

    Inputs
    ----------
    params
        (tuple or array) Tunable parameters of model
    img
        (2d np array) Image to fit


    Optional Inputs
    ----------------
    mask
        (2d np array of bools) Non-zero elements of mask indicate good data 
        which should be included in the fit


    Returns
    ----------
    float
        A positive number measuring of the goodness of fit. Lower values are better
    """

    modelImg = model.get(bbox, params)
    diff = img - modelImg
    diff *= mask

    cost = np.sqrt(np.sum(diff ** 2))
    assert cost > 0
    return cost


class Callback:
    """A useful class for printing intermediate diagnostics

    Usage:
    ---------
    ```
    callback = Callback(model, img, bbox, mask)
    result = spOpt.minimize(..., callback=callback)
    ```
    """
    def __init__(self, model, img, bbox, mask=None):
        self.model = model 
        self.img = img 
        self.bbox = bbox 
        self.mask = mask 
        self.counter = 0 

    def __call__(self, xk):
        score = costFunc(xk, self.model, self.img, self.bbox, self.mask)
        print(f"Itr: {self.counter}. Score {score}: Params: {xk}")
        return False 
