# -*- coding: utf-8 -*-

"""
Utilities for finding and marking bright regions in an image.

An example of use is a grey scale image of some ships in a dark ocean. Use
the algorithms in this module to find the bright ships against the dark background,
then to annotate the original image to show which ships were detected.

The methods here are quite crude. More sophisticated optiosn available at
http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py


Object Detection
-----------------
Two algorithms are provided. Each find a set of contiguous pixels above threshold
(i.e they find a group of pixels that all touch, and are all brighter than a
specified value).

`find_blobs` finds all such bright, contiguous pixels (or blobs),
and assigns a unique number (the *label*) to each blob.

`find_single_blob` finds the blob that contains the supplied pixel. It should
be considerably faster than find_blobs.


Object Marking
--------------
Three options are provided for displaying the masks

`draw_mask` marks the pixels in the mask with the requested colour.

`draw_mask_border` marks only the border pixels in the mask. The border
pixels of the original image are obscured, but the internal pixels are still
visible

 draw_mask_outline` draws a thin line around the outside of the border pixels.
While no pixels in the original image are obscured, this function is considerably
slower than the other two, and may not be suitable for large images.

All draw functions put the origin at bottom left, instead of the matplotlib
default of top right


Glossary
------------
mask
    A boolean array indicating pixels of interest

blob
    A set of contiguous pixels all above some threshold flux

label
    A unique integer identifying a single blob


Created on Wed Feb 27 16:05:30 2019
@author: fergal
"""

from __future__ import print_function
from __future__ import division

from pdb import set_trace as debug
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import copy


def find_blobs(img, threshold):
    """Find contiguous regions of an image brighter than threshold

    Based on the "Connected component labeling algorith of Rosenfeld (1996)
    https://en.wikipedia.org/wiki/Connected-component_labeling
    (see the two pass algorithm)

    Inputs
    ---------
    img
        (2d np array) Input image

    threshold
        (float) Pixels above this threshold are marked.

    Returns
    ----------
    2d np array. Each pixel is assigned an integer. All pixels with the
    same integer are both above threshold and connected to each other.
    Pixels below threshold are set to zero.
    """
    nr, nc = img.shape
    label = np.zeros_like(img, dtype=int)
    labelId = 1

    for i in range(nr):
        for j in range(nc):
            if img[i, j] > threshold:
                if j == 0 or label[i, j-1] == 0:
                    #Create a new labeled region
                    label[i, j] = labelId
                    labelId += 1
                else:
                    #Extend the current labeled region
                    label[i, j] = label[i, j-1]

    #At this point, we have a set of labeled regions, but each
    #is confined to its own column. Now we merged regions that touch
    #By scanning across the rows(instead of down the columns )
    for j in range(nc):
        for i in range(nr-1):
            v1 = label[i, j]
            v2 = label[i+1, j]

            if v1 == v2:
                continue

            if v1 > 0 and v2 > 0:
                idx = label == v2
                label[idx] = v1

    return label


def find_single_blob(img, threshold, col0, row0):
    """Mark the set of contiguous pixels above threshold around the requested position

    Inputs
    ---------
    img
        (2d np array) Input image

    threshold
        (float) Pixels above this threshold are marked.

    col0, row0
        (ints) Location of seed point.

    Returns
    -----------
    A labeled image with zero or one blobs labeled. For consistency
    with `find_blobs` the return value is an integer array.

    """

    region = dict()
    stack = [ np.array((col0, row0)) ]
    delta = np.array( ((-1,0), (1,0), (0,-1), (0,1)) )

    nc, nr = img.shape

    debug()
    while len(stack) > 0:
        cr = stack.pop()

        for d in delta:
            p1 = cr + d

            #Guard against out of bounds
            if p1[0] < 0 or p1[1] < 0:
                continue

            if p1[0] >= nr or p1[1] >= nc:
                continue

            if tuple(p1) not in region:
                if img[ p1[1], p1[0] ] > threshold:
                    stack.append(p1)
                    region[ tuple(p1) ] = True

    mask = np.zeros_like(img, dtype=int)

    for p in region:
        mask[p[1], p[0]] = 1

    return mask


def draw_mask(mask, clr, alpha=1):
    """Draw the pixels in the mask in the requested colour

    Inputs
    ----------
    mask
        (2d boolean array) Pixels marked **True** are drawn. Other pixels
        are marked as completely transparent

    clr
        (matplotlib colour spec) Colour to mask pixels with

    Optional Inputs
    -----------------
    alpha
        (float) Opacity (in range [0,1]) of mask.

    Returns
    ----------
    **None**

    Output
    ----------
    Requested pixels in a pre-plotted image are masked. The image
    itself is not plotted, you have to do that yourself.

    """

    cmap = copy.deepcopy(plt.cm.Reds)
    cmap.set_under('#00FF0000')
    cmap.set_over(clr)
    plt.imshow(mask, origin='bottom', cmap=cmap, alpha=alpha)
    plt.clim(.5, .6)



def draw_mask_border(mask, clr, alpha=1):
    """Mask only the border pixels for a label

    When overlayed on an image, masking only the outline makes it easier
    to see contents of the region under the label

    Inputs
    --------
    label
        (2d np array) Output of `label_image`

    i0
        (int) Label value to outline

    clr
        (matplotlib colourspec) Colour to mark the image with

    Returns
    ----------
    **None**

    Output
    ----------
    The outline of the requested label in a pre-plotted image are masked. The image
    itself is not plotted, you have to do that yourself.

    Note
    --------
    * The border drawn overlaps the pixels on the bottom and left of the
      shape, but is exterior to them on the top and right. This is
      a computation convenience, but results disagree with `draw_mask_outline`

    * Behaviour is undefined when mask touches edge of image
    """

    edges = mask + np.roll(mask, 1, axis=0) == 1
    edges += mask + np.roll(mask, 1, axis=1) == 1

    if np.sum(edges) == 0:
        raise ValueError("No edges found")

#    mask_pixels_in_range(edges, .5, 2, clr)
    draw_mask(edges, clr, alpha)


def draw_mask_outline(mask, **kwargs):
    """Draw a line around a label in a region

    `mask_pixels_in_range` draws a mask on all pixels in a region of an image
    `mask_label_outline` draws a mask on all border pixels in the region
    This function draws a thin line around the border instead.

    It is much better at marking a region without obscuring the region,
    but it is much slower to create a drawing.

    Inputs
    ----------
    label
        (2d np array) Output of `label_image`

    i0
        (int) Label value to outline

    Optional Inputs
    -----------------
    All optional arguments passed to `mpl.collections.LineCollection`
    Common ones are `color` and `lw` for linewidth.

    Returns
    ----------
    **None**

    Output
    ----------
    The border of the requested label in a pre-plotted image are masked. The image
    itself is not plotted, you have to do that yourself.

    Note
    --------
    Behaviour is undefined when mask touches edge of image

    """

    A = mask #Mneumonic

    rollA = np.roll(A, 1, axis=1)
    linelist = []


    #Left side
    rows, cols = np.where(A & ~rollA )
    for col, row in zip(cols, rows):
        line = [(col-.5, row-.5), (col-.5, row+.5)]
        linelist.append(line)

    #Right side
    rows, cols = np.where(rollA & ~A)
    for col, row in zip(cols, rows):
        line = [(col-.5, row-.5), (col-.5, row+.5)]
        linelist.append(line)


    rollA = np.roll(A, 1, axis=0)
    rows, cols = np.where(rollA & ~A)
    for col, row in zip(cols, rows):
        line = [(col-.5, row-.5), (col+.5, row-.5)]
        linelist.append(line)

    rows, cols = np.where(A & ~rollA )
    for col, row in zip(cols, rows):
        line = [(col-.5, row-.5), (col+.5, row-.5)]
        linelist.append(line)

    collection = mpl.collections.LineCollection(linelist, **kwargs)
    plt.gca().add_collection(collection)





def mask_pixels_in_range(img, lwr, upr, clr, alpha=1):
    """Mark pixels in image with values in a given range with the requested colour.

    This is a convenience function for `draw_mask`

    Inputs
    ----------
    img
        (2d array) Image to mask
    lwr, upr
        (floats) Range of pixel values to mask
    clr
        (matplotlib colour spec) Colour to mask pixels with
    alpha
        (float) Opacity (in range [0,1]) of mask.


    Returns
    ----------
    **None**

    Output
    ----------
    Requested pixels in a pre-plotted image are masked. The image
    itself is not plotted, you have to do that yourself.

    """
    mask = get_mask_for_range(img, lwr, upr)
    draw_mask(mask, clr, alpha=alpha)


#
#Convert images, or label images to masks
#
def get_mask_for_label(label, i0):
    mask = np.zeros_like(label, dtype=bool)
    idx = label == i0
    mask[idx] = True
    return mask


def get_mask_for_range(image, lwr, upr):
    idx = (image >= lwr) & (image < upr)
    mask = np.zeros_like(image, dtype=bool)
    mask[idx] = 1
