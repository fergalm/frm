# -*- coding: utf-8 -*-

"""
Draw a street map underneath your plot.

Requires an internet connection

Usage
--------

::
    plt.plot(lng, lat, 'ko')
    mapoverlay.drawMap()

Choices
------
light
    A good, low contrast background style. This is the default
osm
    Default OpenStreetMap style
wikipedia
    Map style used by wikipedia
transparent
    A monochrome map useful of plotting on top of cloropleth maps
planet
    A low res satellite image from planet
sat
    A highres satellite (or aerial) image
satlab
    A labeled satellite iamge

"""
from __future__ import print_function
from __future__ import division

import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import requests
import os


mapStyles ={
             'osm': "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png" ,
             'wikipedia': "https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png" ,
             'light': "http://light_all.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png" ,
             'stamen': "http://a.tile.stamen.com/toner/{z}/{x}/{y}.png",
             'transparent': "http://a.tile.stamen.com/toner/{z}/{x}/{y}.png",
             'planet': "https://tiles.planet.com/basemaps/v1/planet-tiles/global_monthly_2018_05_mosaic/gmap/{z}/{x}/{y}.png?api_key=68d6a04361934ed98effdd27d78df506",
             'mapbox': 'https://api.mapbox.com/styles/v1/mapbox/streets-v10/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibXVsbGFsbHkiLCJhIjoiY2ppdnVvb2k0MnV4MDNrdDhlcW9oc2VhNiJ9.ODtFiPKOQUdJ-IBrO_wVSg',
             'satlab': 'https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v10/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibXVsbGFsbHkiLCJhIjoiY2ppdnVvb2k0MnV4MDNrdDhlcW9oc2VhNiJ9.ODtFiPKOQUdJ-IBrO_wVSg',
             'sat': 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibXVsbGFsbHkiLCJhIjoiY2ppdnVvb2k0MnV4MDNrdDhlcW9oc2VhNiJ9.ODtFiPKOQUdJ-IBrO_wVSg',
           }

#mapbox://styles/mullally/cjivusojc12312rpmvpgty841
mapCopyright={
            'osm': u"Map © OpenStreetMap",
            'transparent': u"Map Design © Stamen by CC-BY. Data © OpenStreetMap",
            'planet': u"Source planet.com",
            'satlab': u"Image from mapbox.com",
            'mapbox': u"Map © mapbox.com",
            'sat': u"Image from mapbox.com",
             }

imageFormat={
        'sat': 'jpg',
        'sat': 'jpg',
        }

maxZoomForStyle={'planet': 15}

def drawMap(style='light', copyright=True, zoom_delta=0):
    """ Download, cache, and display map tiles for the currently defined axis limits

    Optional Inputs
    -----------
    style
        (str) One of the keys of ``mapStyles`` dictionary. Default is osm, which is the default OpenStreetMap
        style

    zoom_delta
        (int) Positive values add more detail to the map, negative values
        reduce the detail (but act to increase the font size of map labels)
    copyright
        (bool) Whether to add a copyright notice in bottom corner of plot

    Notes
    -----------
    * The transparent map is intended to be used to be laid on top of chloropleths (aka heatmaps). For drawing
    bounardaries around objects, osm or wikipedia work better.

    * For most map styles, the map is drawn behind other objects on the page (``zorder := -100``).
    For the transparent style, it has to go on top of any other plot objects, so zorder is set to +100
    """


    if style not in mapStyles.keys():
        raise ValueError("Input style %s not recognised" %(style))

    lng1, lng2, lat1, lat2 = getAxisLimit()

    try:
        maxZoom = maxZoomForStyle[style]
    except:
        maxZoom = 19

    zoom = computeZoomLevel( abs(lng2-lng1), maxZoom ) + zoom_delta

    tileList = computeTileList(lng1, lng2, lat1, lat2, zoom)
    renderTileList(tileList, style, copyright)


def drawPrintableMap(width_in, height_in, dpi=200, style='osm', copyright=True):

    if style not in mapStyles.keys():
        raise ValueError("Input style %s not recognised" %(style))

    lng1, lng2, lat1, lat2 = getAxisLimit()
    nHorzTiles = width_in * dpi / 256.
    nVertTiles = height_in * dpi / 256.
    deltaLong = abs(lng2 - lng1) / nHorzTiles
    deltaLat = abs(lat2 - lat1) / nVertTiles
    deltaAngle = min(deltaLong, deltaLat)

    scale = abs(lng2 - lng1)  + deltaAngle/2.
    scale = max(scale, abs(lat2 - lat1) ) + deltaAngle/2.
    zoom = int(computeZoomLevel(scale))

    tileList = computeTileList(lng1, lng2, lat1, lat2, zoom)
    renderTileList(tileList, style, copyright)



def addScalebar():
    """Draw a scalebar on map to indicate distances in metres or kilometres

    Draws an alternating black and white bar on the map, and labels
    useful distances in metres/kilometres

    The bar has no tunable options. It sizes itself so that

    * It covers approx 1/3 the width of the plot
    * The length is divisible by 8,6,4,2 or 1
    * It is situated in the bottom right corner of the plot

    It automatically adjusts the units between metres and kilometres
    to give meaningful numbers.
    """
    numBars = 4
    earthRadius_m = 6.4e6
    lng1, lng2, lat1, lat2 = np.radians(plt.axis())

    step_m = getScalebarStepsize_metres(numBars, earthRadius_m)
    step_unit, unit = getScalebarUnit(step_m)

    #Compute Right anchor of scale bar, y position, and a stepsize in y
    x0 = np.degrees(lng1 + .9 * (lng2-lng1))
    y0 = np.degrees(lat1 + .1 * (lat2-lat1))
    dy = np.degrees(lat2-lat1)

    step_radlng = step_m / (earthRadius_m * np.cos(lat1))
    step_deglng = np.degrees(step_radlng)

    #Draw bars and add labels
    colourCycle = itertools.cycle('k w'.split())
    ytext = y0 + .014 * dy
    for i in [4,3,2,1]:
        x1 = x0 - i * step_deglng
        x2 = x0 - (i-1) * step_deglng

        label = "%.0f" %( (4-i) * step_unit)
        clr = colourCycle.next()

        plt.plot([x1, x2], [y0, y0], 'k', lw=10, zorder=100)
        plt.plot([x1, x2], [y0, y0], clr, lw=8, zorder=100)
        plt.text(x1, ytext, label, color='k', fontsize=14, ha='left')

    #Add label on right edge of scalebar
    label = "%.0f %s" %( 4 * step_unit, unit)
    handle = plt.text(x0, ytext, label, color='k', fontsize=14, ha='left')

    #Draw a rectangle around the scale bar
    draw_rectangle_around_scalebar(x0, step_deglng, y0, dy, zorder=handle.zorder)


def getScalebarStepsize_metres(numBars, earthRadius_m):
    """Private function of drawScalebar"""
    lng1, lng2, lat1, lat2 = np.radians(plt.axis())

    plotWidth_m = (lng2 - lng1)
    plotWidth_m *= earthRadius_m * np.cos(lat1)

    #Don't want scale bar to cover entire width of page
    plotWidth_m /= 3.

    mag = int(np.log10(plotWidth_m))
    val = plotWidth_m  / 10**mag
    assert val >= 1
    assert val < 10

    print(val, mag)
    allowedSteps = [1,2,4,6,8, 10]
    wh = np.where(allowedSteps > val)[0][0] -1
    step_m = allowedSteps[wh] * 10**mag / float(numBars)
    return step_m


def getScalebarUnit(step_m):
    """Private function of drawScalebar"""
    if step_m >= 1000:
        unit = "km"
        step_unit = step_m / 1e3
    else:
        unit = "m"
        step_unit = step_m

    return step_unit, unit


def draw_rectangle_around_scalebar(x0, step_deglng, y0, dy, zorder=0):
    """Private function of drawScalebar"""
    rx1 = x0 - 4.5*step_deglng
    dx = 7 * step_deglng
    ry1 = y0 - .4*step_deglng
    dy = .08 * dy
    rect = mpatch.Rectangle([rx1, ry1], dx, dy, fc='w', ec='k', zorder=zorder)
    plt.gca().add_patch(rect)


def renderTileList(tileList, style, copyright):
    axl = plt.axis()
    for t in tileList:
        img, extent = getTile(*t, style=style)

        if style == 'transparent':
            img = make_transparent(img)
            zorder = +100
        else:
            zorder = -100

        plt.imshow(img, extent=extent, interpolation='lanczos', zorder=zorder)
#        plt.pause(.001)
    plt.axis(axl)

    if copyright:
        addCopyright(style)


def addCopyright(style='osm'):
    """Adds a copyright message to bottom right of plot.
    If style isn't known, silently defaults to OSM attribution
    """
    if style not in mapCopyright.keys():
        style = 'osm'

    text = mapCopyright[style]

    ax = plt.gca()
    th = plt.text(.98, .02, text, ha='right', transform=ax.transAxes, fontsize=10)
    th.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])


def drawExtent(extent, *args, **kwargs):
    x1,x2,y1,y2 = extent
    plt.plot([x1,x1], [y1,y2], *args, **kwargs)
    plt.plot([x1,x2], [y2,y2], *args, **kwargs)
    plt.plot([x2,x2], [y2,y1], *args, **kwargs)
    plt.plot([x1,x2], [y1,y1], *args, **kwargs)


def getAxisLimit():
    lng1_deg, lng2_deg, lat1_deg, lat2_deg = plt.axis()

    if lng1_deg < -180 or lng2_deg > 360:
        raise ValueError("Longitude must be in range [-180,180)")

    if lng1_deg == lng2_deg:
        raise ValueError("Longitude range is zero")

    if lat1_deg < -90 or lat2_deg > 90:
        raise ValueError("Lat must be in range [-180,180)")

    if lat1_deg == lat2_deg:
        raise ValueError("Latitude range is zero")

    if lng1_deg == 0 and lng2_deg == 1 and lat1_deg == 0 and lat2_deg == 1:
        raise ValueError("Axes are unset")

    return lng1_deg, lng2_deg, lat1_deg, lat2_deg


def computeZoomLevel(deltaLon_deg, maxZoom, offset=+2):
    zoom =  int( np.floor( np.log2(360/deltaLon_deg) ) ) + offset
    zoom = min(zoom, maxZoom)
    return zoom


def computeTileList(lng1_deg, lng2_deg, lat1_deg, lat2_deg, zoom):
    lng = np.array([lng1_deg, lng2_deg])
    lat = np.array([lat1_deg, lat2_deg])

    xrng, yrng = deg2num(lng, lat, zoom)
    xrng.sort()
    yrng.sort()

    out = []
    #@TODO Push this into numpy
    for i in range(xrng[0], xrng[1]+1):
        for j in range(yrng[0], yrng[1]+1):
            out.append( (i,j,zoom))

    return out


def getTile(x, y, zoom, style='osm'):
    url = getTileUrl(x, y, zoom, style)

    try:
        fmt = imageFormat[style]
    except KeyError:
        fmt = 'png'

    home = os.environ['HOME']
    cachePath = os.path.join(home, ".cache", "maptile", "%i" %(zoom))
    cacheFile = os.path.join( cachePath, "tile%s.%s" %(str(hash(url)), fmt) )

    extent = getExtent(x, y, zoom)
    if os.path.exists(cacheFile):
        return plt.imread(cacheFile), extent

    if not os.path.exists(cachePath):
        os.makedirs(cachePath)

    response = requests.get(url, headers={"User-Agent":"MapOverlay fergal.mullally@gmail.com"})
    if response.status_code != requests.codes.ok:
        raise IOError("Query for %s raised error code %i" %(url, response.status_code))

    with open(cacheFile, 'wb') as fp:
        fp.write( response.content)

    return plt.imread(cacheFile), extent




def getTileUrl(x, y, zoom, style):

    params = { 'x':x, 'y':y, 'z':zoom}
    url = mapStyles[style]
    url = url.format(**params)


    return url



def getExtent(x, y, zoom):
    lng0, lat0 = num2deg(x, y, zoom)
    lng1, lat1 = num2deg(x+1, y+1, zoom)

    extent = [lng0, lng1, lat1, lat0]
    return extent


def num2deg(xtile, ytile, zoom):
    """Adapted from http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python"""

    n = 2.0 ** zoom
    lng_deg = xtile / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
    lat_deg = np.degrees(lat_rad)
    return (lng_deg, lat_deg)


def deg2num(lng_deg, lat_deg, zoom):
    """Adapted from http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python"""
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = np.floor(((lng_deg + 180.0) / 360.0 * n)).astype(int)
    ytile = np.floor(((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)).astype(int)
    return xtile, ytile


def make_transparent(img):
    """Make white parts of the map transparent.

    Notes
    -------
    Only tested on the "Stamen Toner" map style
    """
    #Convert to integers for masking
    int_img = (img * 256).astype(int)

    idx = int_img[:,:,0] > 200

    #Create an image with an alpha channel
    if img.shape[-1] == 4:
        im2 = img
    else:
        im2 = np.ones( (256,256,4))
        im2[:,:,:3] = img
    im2[idx,3] = 0   #Make mask transparent

    return im2
