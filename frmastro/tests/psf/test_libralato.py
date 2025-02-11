

from frmastro.psf.libralato import LibralatoMiri
import matplotlib.pyplot as plt 
import frmastro.psf.disp as disp 

#TODO: Make a dummy class that doesn't download

def test_smoke():
    # bbox = Bbox(1000, 1020, 900, 920)

    obj = LibralatoMiri("./")

    plt.clf()
    for i in range(5):
        plt.subplot(1,5,i+1)
        frac=  i/5.
        img = obj.getModelPrfForColRow(1010 + frac, 910)
        assert img.shape == (25,25)
