from frmbase.tstools import find_peaks
import numpy as np
import pytest


def test_find_peaks_slope():
    y = np.arange(10)
    peaks = find_peaks(y)
    assert np.all(peaks == False)


def test_find_peaks_single_peak():
    y = np.ones(10)
    y[5] = 2
    peaks = find_peaks(y)
    assert peaks[5], peaks
    peaks[5] = False 
    assert np.all(peaks == False)

def test_find_peaks_two_peaks():
    y = np.ones(10)
    y[5] = 2
    y[8] = 3
    peaks = find_peaks(y)
    assert peaks[5], peaks    
    assert peaks[8], peaks


def test_find_peaks_peak_at_zero():
    y = np.ones(10)
    y[0] = 5
    peaks = find_peaks(y)
    assert np.all(peaks == False), peaks
    