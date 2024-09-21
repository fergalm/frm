
from frmastro.psf.bbox import Bbox 
import numpy as np

def test_init():

    box = Bbox.fromCRCR(10, 20, 31, 42)
    assert box.shape == (22, 21), box.shape  #Remember shape is (nr, nc)

    box = Bbox.fromCCRR(10, 31, 20, 42)
    assert box.shape == (22, 21), box.shape  #Remember shape is (nr, nc)    

    box = Bbox.fromSize(10, 20, 21, 22)
    assert box.shape == (22, 21), box.shape  #Remember shape is (nr, nc)    
    
    img = np.zeros((22,21))
    box = Bbox.fromImage(img)
    assert box.shape == (22, 21), box.shape  #Remember shape is (nr, nc)    
    assert box.col0 == 0

    print(box) 

def test_getSubImage():

    img = np.arange(20).reshape(4,5)

    bbox = Bbox.fromSize(0, 0, 2, 3)
    bbox.getSubImage(img)
    assert bbox.shape == (3,2)