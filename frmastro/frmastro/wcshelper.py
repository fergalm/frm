from ipdb import set_trace as idebug 
import matplotlib.pyplot as plt 
from astropy.wcs.wcs import WCS as Wcs
import numpy as np 
import frmplots.plots as fplots 

def plotCompassRoseNorthUp(col0, row0, clr='g', length_pix=100, east="left"):
    """Plot a compass rose for the "default" orientation of North up.

    The standard orientation for displaying astronomical images is
    with North pointing  up, and East pointing to the left. If 
    this is the case for your image, you can use this function
    to annotate your image with a compass without mucking around 
    with the WCS. 

    Inputs
    -----------
    col0, row0 (floats)
        Centre of compass rose on the image in units of pixels.
    clr (str)
        Colour of compass rose 
    length_pix (float)
        Length of North arrow in units of pixels 
    east (string)
        if "right", plot the east arrow pointing right instead of 
        left (the default).


    """
    northVec = np.array([0, 1])
    eastVec = np.array([-1, 0])

    if east == "right":
        eastVec *= -1 

    plotArrows(col0, row0, northVec, eastVec, length_pix, clr)

def plotCompassRose(wcs, col0=None, row0=None, clr='g', length_pix=100):
    """
    Plot a compass rose indicating North and East on an image

    Inputs
    ----------
    wcs
        An astropy Wcs object 
    col0, row0 (float)
        The column and row of the centre of the rose. If not 
        specified, the rose is centred at CRPIX from the Wcs
    clr (string)
        Colour to draw the shape.
    length_pix (float)
        Length in pixel of the two arrows 

    Returns
    ------------
    **None**

    Notes
    ---------
    * Assumes the image has not had any rotations applied, that the
    figure axis is aligned with the physical coordinates of the image.
    If you have rotated the image this function doesn't work

    * Unless you specify plt.axis('equal') your compass arrows may 
    not appear to be at right angles. This is an artefact of plotting
    """
    
    col0 = col0 or wcs['CRPIX'][0]
    row0 = row0 or wcs['CRPIX'][1]
    northVec, eastVec = getCompassPoints(wcs)

    plotArrows(col0, row0, northVec, eastVec, length_pix, clr)


def plotArrows(col0, row0, northVec, eastVec, length_pix, clr='g'):
    scale = 1.4
    aprops =dict(
        edgecolor=clr,
        head_width=3,
        head_length=3,
        linewidth=3,
    )

    tprops = dict(
        va="center",
        ha="right",
        color = aprops['edgecolor'],
        fontsize=18,
        fontweight='bold',
        path_effects=fplots.outline('w', lw=3)
    )
    
    #Draw North Vector
    dc = float(length_pix*northVec[0])
    dr = float(length_pix*northVec[1])
    plt.arrow(col0, row0, dc, dr, **aprops)
    plt.text(col0 + scale * dc, row0+ scale * dr, "N ", **tprops)

    #Draw East Vector
    dc = float(length_pix*eastVec[0])
    dr = float(length_pix*eastVec[1])
    plt.arrow(col0, row0, dc, dr, **aprops)
    plt.text(col0 + scale * dc, row0+ scale * dr, "E ", **tprops)

    patch = plt.Circle((col0, row0), radius = (length_pix/5), fc="None", ec=clr, lw=aprops['linewidth'])
    plt.gca().add_patch(patch)
    patch = plt.Circle((col0, row0), radius = (length_pix/20), fc=clr)
    plt.gca().add_patch(patch)


def getImageRotationFromNorth(wcs):
    """
    The the angle of rotation of an image.

    Inputs
    ------------
    wcs
        An astropy Wcs object 

    Returns
    ---------
    The rotation angle of the wcs in degrees west of North.
    To rotate the image so North points up use, e.g 

    ```
    angle_deg_won = getImageRotationFromNorth(wcs)
    scipy.ndimage.rotate(image, -angle_deg_won)
    ```
    """
    northVec, _ = getCompassPoints(wcs)
    #Get rotation angle north of east
    ang_rad_noe = np.arctan2(northVec[1], northVec[0])
    ang_deg_noe = np.degrees(ang_rad_noe)

    #I want zero degrees to mean north
    ang_deg_won = ang_deg_noe - 90
    return ang_deg_won[0]



def getCompassPoints(wcs):
    """Get vectors pointing North and East in pixel space from a WCS 

    Inputs
    -----------
    wcs
        Either 
        * A Wcs object. 
        * A fits header. Must contain the keys CD1_1, CD1_2, CD2_1, CD2_2 

    Returns
    ----------
    northVec  (2x1 numpy array)
        A vector, 1 pixel long indicating the direction of north. The
        contents are  `[ [numPixelsOfIncreasingColumn], [numPixelsOfIncreasingRow]]`
    eastVec
        Same as `northVec`, but for the easterly direction

    Note 
    ----------
    The direction of the east vector may be wrong for some images. For 
    most astro images east is left of North. If the determinant of the 
    CD matrix is+ve then east is probably the other direction. But 
    I don't have good data to test with
    """
    try:
        cdMat = getCdFromFitsHdr(wcs)
    except TypeError:
        cdMat = getCdFromWcs(wcs)
                  
    up = np.array([0,1]).reshape(2,1)
    right = np.array([1,0]).reshape(2,1)

    northVec = cdMat @ up
    eastVec = cdMat @ right

    #Normalise vectors so each is 1 pixel long
    northVec /= np.linalg.norm(northVec)
    eastVec /= np.linalg.norm(eastVec)
    return northVec, eastVec


def getCdFromWcs(wcs):
    """Not tested"""
    try:
        cdMat = wcs.wcs.cd 
    except (TypeError, AttributeError):
        pcMat = wcs.wcs.pc
        cDelt = wcs.wcs.cdelt.reshape(1,2)
        cDelt = np.array([
            [ cDelt[0,0], 0],
            [0, cDelt[0,1]],
        ])
        cdMat = pcMat @ cDelt

    return cdMat

def getCdFromFitsHdr(hdr):

    if 'CD1_1' in hdr:
        cdMat = np.zeros((2,2))
        cdMat[0,0] = hdr['CD1_1']
        cdMat[0,1] = hdr['CD1_2']
        cdMat[1,0] = hdr['CD2_1']
        cdMat[1,1] = hdr['CD2_2']
    elif 'PC1_1' in hdr:
        pcMat = np.zeros((2,2))
        pcMat[0,0] = hdr['PC1_1']
        pcMat[0,1] = hdr['PC1_2']
        pcMat[1,0] = hdr['PC2_1']
        pcMat[1,1] = hdr['PC2_2']

        cDelt = np.array([
            [ hdr['CDELT1'], 0],
            [0, hdr['CDELT2']],
        ])
        cdMat = cDelt @ pcMat 
    else:
        raise KeyError("Can't find CD or PC elements in header")

    assert cdMat.shape == (2,2)
    return cdMat



def test_plotCompassRose():

    plt.clf()
    plt.axis([0, 512, 0, 512])

    wcs = {
        'CD1_1' : 8.6732557453218E-06,
        'CD1_2' : -2.9737346174155E-05 ,
        'CD2_1' : -2.9444839887394E-05,
        'CD2_2' : -8.4994789836792E-06,
    }        

    plotCompassRose(wcs, 256, 256)

# def test_getCompassRose():

#     cdMat = np.zeros((2,2))
#     cdMat[0,0] = 8.6732557453218E-06
#     cdMat[0,1] = -2.9737346174155E-05 
#     cdMat[1,0] = -2.9444839887394E-05
#     cdMat[1,1] = -8.4994789836792E-06

#     wcs = {
#         'CD1_1' : 8.6732557453218E-06,
#         'CD1_2' : -2.9737346174155E-05 ,
#         'CD2_1' : -2.9444839887394E-05,
#         'CD2_2' : -8.4994789836792E-06,
#     }        
#     getCompassRose(wcs)

