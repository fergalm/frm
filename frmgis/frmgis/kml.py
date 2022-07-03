import matplotlib.pyplot as plt
import osgeo.ogr as ogr
import numpy as np
import fastkml
import re

"""
Convert list of GDALs to matplotlib shapes
cloropleths

Multipolygon fails

"""


def load(fn):
    """Load a kml file from disk"""
    with open(fn) as fp:
        text = fp.read()

    obj = fastkml.kml.KML()
    obj.from_string(text)
    return obj


def gdalListFromKml(obj):
    if hasattr(obj, "features"):
        featureList = list(obj.features())
        out = map(gdalListFromKml, featureList)
        return out
    else:
        return ogr.CreateGeometryFromWkt(obj.geometry.to_wkt())


def view(obj, level=0, maxLeaf=10, keys=None):
    """Print a summary of the kml file

    Inputs:
    ----------
    obj
        a fastkml.kml.KML() object

    Optional Inputs
    --------------
    maxLeaf
        (int) Show no more than this many shapes for each folder

    level
        (int) Used to control print formatting. Not to be set by end user

    keys
        (list of strings) extract and prrint this property of each shape

    Returns
    -------------
    **None**

    Output
    ----------
    Prints a summary of the file to screen
    """

    levelStr = "  "
    if hasattr(obj, "features"):
        description = get_metadata(obj, keys, recurse=False)
        print( levelStr*level, obj, description)

        featureList = list(obj.features())
        numToPrint = min(len(featureList), maxLeaf)
        for f in featureList[:numToPrint]:
            view(f, level+1, maxLeaf=maxLeaf, keys=keys)

        if numToPrint == maxLeaf:
            print( levelStr * (level+1), "... total of %i shapes" %(len(featureList)))
    else:
#        import pdb; pdb.set_trace()
        description = []
        if keys is not None:
            description = get_metadata(obj, keys)

        print( levelStr*level, obj.geometry.geom_type, \
            " ".join(description).encode('utf-8'))
        #print levelStr*level, "Leaf node found"


def get_metadata(obj, keys, recurse=True):
    """

    keys
        Iterable.
    """

    if keys is None:
        return []

    if isinstance(keys, str):
        keys = [keys]

    if hasattr(obj, "features") and recurse is True:
        featureList = list(obj.features())
        out = map(lambda x: get_metadata(x, keys), featureList)
        return out
    else:
        text = obj.to_string().split('\n')

        description = []
        for k in keys:
            pattern = "<.*%s.*>(.*)<" %(k)
            lines  = map( lambda x: re.search(pattern, x), text)
            lines = filter(lambda x: x is not None, lines)

            if len(lines) > 0:
                description.append(lines[0].group(1))
            else:
                description.append(" ")
        return description


