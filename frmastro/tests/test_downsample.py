
from frmastro.downsample import downsample_2darray
import numpy as np

def test_downsample_2darray():

    arr = np.arange(24).reshape(6,4)  #6 rows, 4 columns 
    arr2 = downsample_2darray(arr, 3, 2)
    assert arr2.shape == (2,2)
    assert arr2[0,0] == 0 + 1 + 4 +5 + 8 + 9, arr2

    arr = np.arange(36).reshape(9,4)  #9 rows, 4 columns 
    arr2 = downsample_2darray(arr, 3, 2)
    assert arr2.shape == (3,2)    
    assert arr2[0,0] == 0 + 1 + 4 +5 + 8 + 9
