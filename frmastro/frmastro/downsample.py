
import numpy as np


def downsample_2darray(arr, sub_pixels_per_row, sub_pixels_per_col):
    """Downsample a 2d array to a smaller array.

    Suppose you have an nxn array, and you want to bin it 2x2 to 
    make a smaller array. You produce the smaller array by::

        >>> new_array = downsample_2darray(arr, 2, 2)

    This is useful when you have an oversampled image that you want
    to resample to, eg, pull out faint features.
    
    Inputs
    ---------
    arr
        (2d np array) Large arry to downsample 
    sub_pixels_per_row
        (int) Number of pixels to bin by in the row direction
    sub_pixels_per_col
        (int) Number of pixels to bin by in the row direction


    Returns
    -------
    A 2d numpy array. If the input array has shape (nr,nc), the output
    shape is (nr/sub_pixels_per_row, nc/sub_pixels_per_col).

    Notes
    ------------
    * The input array shape must be a multiple of the sub pixels. For example,
    if the number of rows in the input array is 12, `sub_pixels_per_row` can
    be 1,2,3,4 or 6, but not 5.

    * Based on a Stack Overflow solution
    https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array

    """
    nr, nc = arr.shape
    num_col_out = nc // sub_pixels_per_col 
    num_row_out = nr // sub_pixels_per_row
    
    assert num_col_out * sub_pixels_per_col == nc 
    assert num_row_out * sub_pixels_per_row == nr
    temp = arr.reshape(num_row_out, sub_pixels_per_row, num_col_out,  sub_pixels_per_col)
    temp = temp.sum(axis=3).sum(axis=1)
    
    assert temp.shape == (num_row_out, num_col_out)
    return temp 
