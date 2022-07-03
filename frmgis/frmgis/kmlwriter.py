"""
Created on Fri Jan 13 09:45:09 2017

Note, for 90% of your use cases use KmlWriter instead of BaseKmlWriter. It will ensure your
polygons are drawn as hollow red shapes instead of opaque white ones

@author: fergal
"""
from __future__ import print_function
from __future__ import division


from frmgis.anygeom import AnyGeom
import datetime
import inspect
import os

class BaseKmlWriter(object):
    """Write a KML file

    ``AnyGeom`` can convert many kinds of location information to a kml snippet. This
    class wraps a set of those in the approprite kml to make a readable file.

    It's still very bare bones, with no support for layers and limited support for styles.

    Usage
    ------------
    Example::

        obj = KmlWriter("file.kml")
        obj.add_style('red', default_polygon_style)
        obj.add( shape, style='red')
        obj.write()


    @TODO
    ---------
    * add() method should replace illegal characters (like '&' from
    text before adding to the object
    """

    def __init__(self, filename, comment=None):
        self.filename  = filename
        self.header = []
        self.text = []
        self.footer = []

        self._add_header(comment)


    def _add_header(self, comment):
        self.header.append('<?xml version="1.0" encoding="UTF-8"?>')
        self.header.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        self._add_watermark(comment)
        self.header.append('<Document>')

    def _add_watermark(self, text=None):
        """Add a comment indicating the source code generating the file
        """

        frame = inspect.currentframe().f_back.f_back.f_back.f_back

        self.add_comment("User: %s" %(os.environ['USER']))
        self.add_comment("Date: %s" %(datetime.datetime.now()))

        filename = frame.f_globals['__file__']
        self.add_comment("File: %s" %(filename))

        frameInfo = inspect.getframeinfo(frame)
        funcname = frameInfo[2]
        lineno = frameInfo[1]
        self.add_comment("Function: %s:%i" %(funcname, lineno))

        if text is not None:
            self.add_comment(text)



    def add_comment(self, text):
        self.header.append("<!-- %s -->" %(text))

    def add_style(self, name, props):
        """Add a stylesheet to the kml file

        Inputs
        ---------
        name
            (string) Name of style sheet
        props
            (dict) Dictionary of properties. See ``default_polygon_style`` for more info

        """
        text = []
        text.append("<Style id='%s'>" %(name))
        text.extend( dict_to_style_kml(props) )
        text.append("</Style>")
        self.header.extend(text)


    def _add_footer(self):
        self.footer.append('</Document>\n</kml>')


    def add(self, obj, name, gtype=None, metadata=None, style_name=None):
        """Add a place to the kml file

        Inputs
        ----------
        obj
            (a shape object) This is any object that can be parsed by ``AnyGeom`` into kml

        Optional Inputs
        ---------------
        gtype
            (string) If obj is a numpy array, use gtype to specify the shape type (e.g POINT, POLYGON
            etc.)
        metadata
            (dict) Any additional metadata to add to the object
        style_name
            (str) Name of style to attach to this object. Styles must be specified with ``add_style``
        """
        text = AnyGeom(obj, gtype, metadata).as_kml()
        text = text.split('\n')
        # text.insert(1, "  <name>%s</name>" %(name))

        if style_name is not None:
            text.insert(1, "  <styleUrl>#%s</styleUrl>" %(style_name))

        text = "\n".join(text)
        self.text.append(text)


    def write(self):
        self._add_footer()

        with open(self.filename, 'w') as fp:
            fp.write("\n".join( self.header) )
            fp.write("\n".join( self.text) )
            fp.write("\n".join( self.footer) )




def dict_to_style_kml(props):
    """Convert a dictioanry to kml in the format used to store style information"""

    text = []
    for k in props.keys():
        text.append("<%s>" %(k))
        value = props[k]
        if isinstance(value, dict):
            text.extend( dict_to_style_kml(value) )
        else:
            text.append(str(value))
        text.append("</%s>" %(k))

    return text



def default_polygon_style():
    """Create a dictionary of properties that can be parsed by ``dict_to_style_kml()``

    Returns
    ------------
    A dictionary

    Notes
    ---------
    * Based on https://developers.google.com/kml/documentation/kml_tut#geometrystyles
    * Format for colour codes is AABBGGRR, or the reverse of HTML and matplotlib codes AA is the alpha.
      00 means transparent and FF means fully opaque
    """
    props = dict()
    props['LineStyle'] = {'width': '4.0', 'color': 'FF0000FF'}
    props['PolyStyle'] = {'color': '00ff0000', 'fill': '0'}
    return props





class KmlWriter(BaseKmlWriter):
    """A special case of BaseKmlWriter that simplifies 90% of my usecase by setting a default style"""

    def __init__(self, filename, comment=None):
        BaseKmlWriter.__init__(self, filename, comment)
        self.add_style('red', default_polygon_style())


    def add(self, obj, name, gtype=None, metadata=None, style_name=None):
        if style_name is None:
            style_name = 'red'

#        import pdb; pdb.set_trace()
        BaseKmlWriter.add(self, obj, name, gtype, metadata, style_name)
