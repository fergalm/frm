

"""
This module contains constants that map to official Orbital Insight colors.

Copied from OFL.base.oi_colors. I think the orriginal was by B. Bakenko

The color palette echoes the front end palette found in orbital-light.css.
It also registers two pyplot colormaps that you can use
(simply importing this module into your code makes those colormaps available).
For example:

>>> from matplotlib import pyplot as plt
>>> import OFL.base.oi_colors
>>> plt.imshow((np.random.rand(10,10)*255).astype(np.uint8), cmap='oi', interpolation='none')
>>> plt.colorbar()

>>> plt.figure()
>>> cm = plt.get_cmap('oi_ext')
>>> for i in range(8):
>>>     plt.plot(np.cos(np.linspace(0,np.pi,100)*(i+1)), color=cm(i/8.0), lw=2)
>>> plt.legend([str(i+1) for i in range(8)])

Of the many colormaps available, two are most useful
* 'oi' -- this sequential colormap interpolates between the 3 dark colors of the OI palette.
   Use this map when you're using color to indicate magnitude (e.g dark blue
   for small values, green for large values etc.)
* 'oi_div' -- this diverging colormap is useful to highlight values that diverge
   from some central value. Green points are above average, blue are below,
   and grey are close to average.

There are a number of other colormaps

* 'oi_r' -- Reverse of oi
* 'oi_div_r' -- Reverse of oi_div
* 'oi_listed' -- this colormap cycles over the 3 dark colors of the OI palette; if you are
    plotting more than 3 things, you are advised to use the 'oi_listed_ext' or 'oi' colormap instead
* 'oi_listed_ext' -- this colormap cycles over the extended colors of the OI palette (7 colors);
    if you are plotting more than 7 things, you are advised to use 'oi' or 'oi_extended' instead
* 'oi_ext' -- this colormap interpolates between the extended OI palette. Use
  this colormap for catagorical plots, where the order of the colours doesn't
  matter (e.g one color for cars, another for trucks etc.)
"""

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import pyplot as plt
import numpy as np

def rgb_to_hex(rgb):
    return '#' + ''.join(['%0.2X' % (_) for _ in rgb])

# brand colors
OI_GREEN_RGB        = (135, 224, 0)
OI_LIGHT_BLUE_RGB   = (0, 150, 189)
OI_DARK_BLUE_RGB    = (15, 56, 77)
# extended colors
OI_LIGHT_GRAY_RGB   = (242, 242, 242)
OI_GRAY_RGB         = (46, 46, 52)
OI_ORANGE_RGB       = (230, 123, 23)
OI_PURPLE_RGB       = (108, 16, 179)
OI_RED_RGB          = (191, 5, 17)
OI_ROYAL_BLUE_RGB   = (32, 86, 189)
OI_YELLOW_RGB       = (212, 212, 15)
OI_MAGENTA_RGB      = (204, 20, 158)

# brand colors
OI_GREEN_HEX        = rgb_to_hex(OI_GREEN_RGB)
OI_LIGHT_BLUE_HEX   = rgb_to_hex(OI_LIGHT_BLUE_RGB)
OI_DARK_BLUE_HEX    = rgb_to_hex(OI_DARK_BLUE_RGB)
# extended colors
OI_LIGHT_GRAY_HEX   = rgb_to_hex(OI_LIGHT_GRAY_RGB)
OI_GRAY_HEX         = rgb_to_hex(OI_GRAY_RGB)
OI_ORANGE_HEX       = rgb_to_hex(OI_ORANGE_RGB)
OI_PURPLE_HEX       = rgb_to_hex(OI_PURPLE_RGB)
OI_RED_HEX          = rgb_to_hex(OI_RED_RGB)
OI_ROYAL_BLUE_HEX   = rgb_to_hex(OI_ROYAL_BLUE_RGB)
OI_YELLOW_HEX       = rgb_to_hex(OI_YELLOW_RGB)
OI_MAGENTA_HEX      = rgb_to_hex(OI_MAGENTA_RGB)
# color scheme on frontend portal
# oi portal colors
OI_CHART_GREEN_HEX = '#87E000'
OI_CHART_DARK_GREEN_HEX = '#31b000'
OI_CHART_BLUE_HEX = '#027cc2'
OI_CHART_LIGHTBLUE_HEX = '#0a9094'
OI_CHART_PURLE_HEX = '#9966ff'
OI_CHART_RED_HEX = '#F15B5A'
OI_CHART_ORANGE_HEX = '#FA7300'
OI_CHART_YELLOW_HEX = '#FFCD00'
OI_CHART_DARKGRAY_HEX = '#2E2E34'
OI_CHART_BLUE_GRAY_HEX ='#84ACB6'
OI_CHART_LIGHT_GRAY_HEX ='#cccccc'

official_colors = np.array([
    OI_DARK_BLUE_RGB,
    OI_LIGHT_BLUE_RGB,
    OI_GREEN_RGB,
    ])/255.0


#End piece colors are not part of the official
#OI colorset, but give a colormap that's roughly
#symettric in luminance
diverging_colors = np.array([
    (0, 78, 90),
    OI_LIGHT_BLUE_RGB,
    OI_LIGHT_GRAY_RGB,
    OI_GREEN_RGB,
    (0, 128, 0),
    ])/255.0

extended_colors = np.array([
    OI_DARK_BLUE_RGB,
    OI_LIGHT_BLUE_RGB,
    OI_GREEN_RGB,
    OI_YELLOW_RGB,
    OI_ORANGE_RGB,
    OI_RED_RGB,
    OI_MAGENTA_RGB,
    OI_PURPLE_RGB,
    ])/255.0


#Linear colormap
oi_cmap = LinearSegmentedColormap.from_list(
    colors=official_colors,
    name='oi')
plt.register_cmap(cmap=oi_cmap)

oi_cmap_r = LinearSegmentedColormap.from_list(
    colors=official_colors[::-1],
    name='oi_r')
plt.register_cmap(cmap=oi_cmap_r)


#Diverging colormap
oi_div_cmap = LinearSegmentedColormap.from_list(
    colors=diverging_colors,
    name='oi_div')
plt.register_cmap(cmap=oi_div_cmap)

oi_div_cmap_r = LinearSegmentedColormap.from_list(
    colors=diverging_colors[::-1],
    name='oi_div_r')
plt.register_cmap(cmap=oi_div_cmap_r)


#Other colormaps
oi_listed_cmap = ListedColormap(
    colors=official_colors,
    name='oi_listed',
    N=len(official_colors))
plt.register_cmap(cmap=oi_listed_cmap)


oi_listed_ext_cmap = ListedColormap(
    colors=extended_colors,
    name='oi_listed_ext',
    N=len(extended_colors))
plt.register_cmap(cmap=oi_listed_ext_cmap)

oi_ext_cmap = LinearSegmentedColormap.from_list(
    colors=extended_colors,
    name='oi_ext')
plt.register_cmap(cmap=oi_ext_cmap)
