# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as mp
import matplotlib as mpl
import numpy as np

import datetime as dt
import inspect
import os

"""
Site-customisation 

Imports the Orbital Insight style file, and provides functions for
annotating a plot. Functions include

add_minor_ticks()
    Add minor ticks to plot axes, which matplotlib does not do by default
    unless the plot is log scaled.

add_watermark()
    Add a small identifying string to the plot indicating the plot
    creator, date, and source file

mark_as_draft()
    Write Draft is big grey letters over the plot

"""

#Windows specific
if 'USER' not in os.environ:
    os.environ['USER'] = 'fergal'

#Auto import the style file. Is this a good idea?
try:
    fname = os.path.join(mpl.get_configdir(), 'fergal.mplstyle')
    mpl.style.use(fname)
except IOError:
    print ("WARN: Fergal's style file not found")
    print ("Config dir is %s" %(mpl.get_configdir()))


def add_minor_ticks():
    """Add minor ticks to axes"""
    ax = mp.gca()
    for a in [ax.xaxis, ax.yaxis]:
        if a.get_scale() == 'linear':
            a.set_minor_locator(mpl.ticker.AutoMinorLocator())

        a.set_tick_params(which="major", length=8, width=1, color='k')
        a.set_tick_params(which="minor", length=4, width=1, color='k')


def remove_minor_ticks():
    ax = mp.gca()
    for a in [ax.xaxis, ax.yaxis]:
        a.set_minor_locator(mpl.ticker.NullLocator())


def add_watermark(level=0, loc='right'):
    """Put an mark on the bottom right of a figure identifing
    plot creator, date, and source file.

    Optional Inputs
    -----------------
    level
        (int) How many levels up the calling stack is the function whose
        name should be included in the watermark test. Default is the function
        that calls add_watermark()

        Example::

            foo():
                bar():
                    add_watermark(0)  #Watermark says $USER code.py:bar ...
                    add_watermark(1)  #Watermark says $USER code.py:foo ...
    """

    text = create_watermark_text(level+1)
    ax = mp.gca()
    f = mp.gcf()

    if loc == 'right':
        ax.text(0.93, 0.88, text, rotation=90, size=8, transform=f.transFigure, ha="right", va="top")
    elif loc == 'bottom':
        ax.text(0.95, 0.01, text, size=8, transform=f.transFigure, ha="right")
    elif loc == 'top':
        ax.text(0.95, 0.96, text, size=8, transform=f.transFigure, ha="right")
    else:
        raise ValueError("loc should be one of 'right', 'top', or 'bottom'")

    
def create_watermark_text(level=0):
    """Create the text string for the watermark. See add_watermark for details"""
    user = os.environ['USER']

    full_fn = '/.'
    try:
        frame = inspect.currentframe().f_back
        for i in range(level):
            try:
                frame = frame.f_back
            except AttributeError:
                raise AttributeError("Requested level (%i) exceeds stack depth (%i)" %(level, i))

        full_fn = frame.f_globals['__file__']
        fn = os.path.basename(full_fn)

        func = frame.f_code.co_name
        fn = "%s:%s" %(fn, func)
    except KeyError:
        fn = "[Shell]"
    time = dt.datetime.strftime( dt.datetime.now(), '%Y-%m-%d %H:%M')

    try:
        import frmbase.meta as meta
        git_meta = meta.get_git_info(full_fn)
        git_hash = git_meta['__git_branch_commit_sha__']
        git_status = git_meta['__git_commit_status__'][0] #One char
        git_hash = "%s (%s)" %(git_hash, git_status)
    except (ImportError, KeyError):
        git_hash = ""

    text = "%s %s %s %s" % (user, fn, time, git_hash)
    return text 


def mark_as_draft(text="Draft"):
    """Write the word draft on a plot in big letters

    Optional Inputs:
    --------------
    text
        Text to write

    """
    ax = mp.gca()
    mp.text(.5, .5, text, va="center", ha="center", fontsize=80, color='k',
            alpha=.4, transform=ax.transAxes, rotation=45)
