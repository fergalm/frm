

from frmastro.psf.libralato import LibralatoMiri
import matplotlib.pyplot as plt 
import frmastro.psf.disp as disp 


def test_smoke():
    # bbox = Bbox(1000, 1020, 900, 920)

    obj = LibralatoMiri("./")

    plt.clf()
    for i in range(5):
        plt.subplot(1,5,i+1)
        frac=  i/5.
        img = obj.getInterpRegPrfForColRow(1010 + frac, 910)
        # disp.plotImage(img)
        # plt.pause(1)

        # def getInterpRegPrfForColRow(self, col:float, row:float)-> InterpRegImage:
