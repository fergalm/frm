from ipdb import set_trace as idebug
import numpy as np 
#from frmgis.anygeom import AnyGeom, ogrGeometryToNestedList
import frmgis.anygeom
from pprint import pprint 


from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

from bokeh.models import ColumnDataSource, Grid, LinearAxis, MultiPolygons, Plot


def test_simple_polygon():

    wkt = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"

    geom = frmgis.anygeom.AnyGeom(wkt).as_geometry()
    xs = ogrGeometryToNestedList(geom, 0)


    assert len(xs) == 1  #One poly in multi 
    assert len(xs[0]) == 1  #Poly with no holes 
    assert len(xs[0][0]) == 5  #Is a square
    assert xs[0][0][0] == 30
    pprint(xs)

    ys = ogrGeometryToNestedList(geom, 1)
    source = ColumnDataSource(dict(xs=[xs], ys=[ys]))
    plot(source)


def test_polygon_with_hole():
    wkt = "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"

    geom = frmgis.anygeom.AnyGeom(wkt).as_geometry()
    xs = ogrGeometryToNestedList(geom, 0)

    assert len(xs) == 1  #One poly in multi 
    assert len(xs[0]) == 2  #Poly with no holes 
    assert len(xs[0][0]) == 5  #Is a square
    assert len(xs[0][1]) == 4  #The hole is a triangle
    pprint(xs)


def test_simple_multipolygon():
    wkt = "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)),((15 5, 40 10, 10 20, 5 10, 15 5)))"

    geom = frmgis.anygeom.AnyGeom(wkt).as_geometry()
    xs = ogrGeometryToNestedList(geom, 0)
    print(xs)
    assert len(xs) == 2  #Two poly in multi 
    assert len(xs[0]) == 1  #Poly with no holes 
    assert len(xs[1]) == 1  #Poly with no holes 

    assert len(xs[0][0]) == 4  #Is a triangle
    assert len(xs[1][0]) == 5  #IS a square


    # assert len(xs[0][0]) == 1  #Is simple polygon
    # assert len(xs[1][0]) == 1  #Is a simple polygon

    # assert len(xs[0][0][0]) == 4  #Is a triangle
    # assert len(xs[1][0][0]) == 5  #IS a square


def test_multipolygon_with_hole():
    """    
        [
            [
                [40.0, 20.0, 45.0, 40.0]
            ], 
            [
                [20.0, 10.0, 10.0, 30.0, 45.0, 20.0], 
                [30.0, 20.0, 20.0, 30.0]
            ]
        ]
    """
    wkt = "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)),((20 35, 10 30, 10 10, 30 5, 45 20, 20 35),(30 20, 20 15, 20 25, 30 20)))"

    geom = frmgis.anygeom.AnyGeom(wkt).as_geometry()
    xs = ogrGeometryToNestedList(geom, 0)
    print(xs)

    assert len(xs) == 2  #Two poly in multi 
    assert len(xs[0]) == 1  #Poly with no holes 
    assert len(xs[1]) == 2  #Poly with one holes 
    
    assert len(xs[0][0]) == 4  #Is a triangle
    assert len(xs[1][0]) == 6  #2nd poly Is a pentago
    assert len(xs[1][1]) == 4  #Hole in 2nd poly Is a triangle

    ys = ogrGeometryToNestedList(geom, 1)
    source = ColumnDataSource(dict(xs=[xs], ys=[ys]))
    plot(source)



    # pprint(ys)

# def test_simple_multipolygon():


#     poly = np.zeros((5,2))
#     poly[:,0] =  [, [1, 1, 2, 2, 1]
#     poly[:,1] = [1, 2, 2, 1, 1]

#     geom = frmgis.anygeom.AnyGeom(poly, 'polygon').as_geometry()
    
#     xs = ogrGeometryToNestedList(geom, 0)
#     ys = ogrGeometryToNestedList(geom, 1)
#     pprint(xs)
#     pprint(ys)


#     source = ColumnDataSource(dict(xs=[[xs]], ys=[[ys]]))
#     plot(source)


def plot(source):
    plot = Plot(
        title="Test Simple Polygon", width=600, height=600,
        min_border=1)

    glyph = MultiPolygons(xs="xs", ys="ys", line_width=2)
    plot.add_glyph(source, glyph)

    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    # curdoc().add_root(plot)

    show(plot)


def ogrGeometryToNestedList(geometry, dim):
    out = []

    name = geometry.GetGeometryName()
    print(f"Geom of type {name}")
    if name == 'MULTIPOLYGON':
        print("is multi")
        for i in range(geometry.GetGeometryCount()):
            print(f"Multi subpoly {i}")
            poly = geometry.GetGeometryRef(i)
            out.extend(ogrGeometryToNestedList(poly, dim))
    elif name == 'POLYGON':
        print("is poly")
        ringlist = []
        for i in range(geometry.GetGeometryCount()):
            print(f"poly ring {i}")
            ring = geometry.GetGeometryRef(i)
            points = np.atleast_2d(ring.GetPoints())
            points = points[:, dim]
            ringlist.append( points.tolist())
        # idebug()
        out.append(ringlist)
    else:
        raise ConversionError("Bokeh only supports polygons and multipolygons")
    
    return out 

