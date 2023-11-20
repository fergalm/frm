import numpy as np 
from frmgis.anygeom import AnyGeom 
from pprint import pprint 

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

from bokeh.models import ColumnDataSource, Grid, LinearAxis, MultiPolygons, Plot


def test_simple_polygon():

    wkt = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"
    xs, ys = AnyGeom(wkt).as_bokeh()
    source1 = ColumnDataSource(dict(xs=[xs], ys=[ys], pcolor=['green']))

    plot(source1)


def test_polygon_with_hole():
    wkt = "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"
    xs, ys = AnyGeom(wkt).as_bokeh()
    source1 = ColumnDataSource(dict(xs=[xs], ys=[ys], pcolor=['red']  ))

    plot(source1)

def test_polygon_list():
    wkt = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"
    x1, y1 = AnyGeom(wkt).as_bokeh()

    wkt = "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"
    x2, y2 = AnyGeom(wkt).as_bokeh()

    clr = ['green', 'red']
    src= ColumnDataSource(dict(xs=[x1, x2], ys=[y1, y2], pcolor=clr))
    plot(src)


def plot(source):
    plot = Plot(
        title="Test Simple Polygon", width=600, height=600,
        min_border=1)

    glyph = MultiPolygons(xs="xs", ys="ys", line_width=2, fill_color="pcolor")
    plot.add_glyph(source, glyph)

    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
    show(plot)
