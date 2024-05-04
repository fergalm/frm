# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:06:00 2020

@author: fergal
"""

from ipdb import set_trace as idebug
from tqdm import tqdm
import pandas as pd
import numpy as np

from frmbase.support import lmap
from . import plots as fgplots
from .anygeom import AnyGeom
import rtree


class GeomCollection():
    """Quick and dirty code to figure out which of a set of geometries
    contains the requested geometry.

    Ensures a requested geom is considered contained even if they
    share an edge.

    Assumes none of the input geometries overlap (e.g they are
    precincts)
    """
    def __init__(self, geom_df, name_col="name", geom_col="geom"):
        self.size = len(geom_df)
        self.geom_df = geom_df.copy()
        self.name_col = name_col
        self.geom_col = geom_col

        #Ensure the geoms are geometries, not, eg WKTs
        self.geom_df[self.geom_col] = lmap(lambda x: AnyGeom(x).as_geometry(), self.geom_df[geom_col])
        self.geom_tree = self.create_tree()

    def create_tree(self):
        tree = rtree.index.Index(interleaved=False)
        for i, row in self.geom_df.iterrows():
            geom  = AnyGeom(row[self.geom_col]).as_geometry()
            env = geom.GetEnvelope()
            tree.insert(i, env)
        return tree

    def measure_overlap(self, shape):
        """Measure overlap between each geometry in the collection and the input shape

        Inputs
        -----------
        shape
            Any AnyGeom compatible shape
        
        Returns
        ---------
        A Dataframe. The columns are 
            name
                Names of the geometries in the collection
            frac
                Fraction of that geometry that is in the shape 

        Note that the some of the fractions need not add up to 1 of more 
        than one geometry is inside the supplied shape 
        """

        df = pd.DataFrame()
        df['frac'] = self._compute_overlap(shape)
        df['name'] = self.geom_df[self.name_col]
        return df

    def measure_overlap_with_df(self, df, name_col="Name", geom_col="geom"):
        """Measure overlap between each geometry in the collection and 
            every geometry in th dataframe

        Inputs
        -----------
        df
            pd.DataFrame. 
        name_col
            (str) column in df that gives the identifing name of the shape.
            Value in this column should typically be unique
        geom_col
            (str) column in df that supplies the geometries of the shapes.
            These can be in any AnyGeom compatible format
        
        Returns
        ---------
        A Dataframe. The index are the names of the geometries in the collection.
        Each shape in the input dataframe is given its own column in the output,
        The value of each element is the fraction of the given geometry that 
        is inside that shape. For a collection geometry that is entirely 
        inside the union of of shapes in the dataframe, the sum of each row is 1.
        There is such rule for the columns
        
        Notes
        ----------
        This is not a fast function
        """

        names = df[name_col]
        out = pd.DataFrame(columns=names)

        elts = list(zip(df[name_col], df[geom_col]))
        for name, shape in tqdm(elts):
            overlap = self._compute_overlap(shape)
            out[name] = overlap 
        out.index = self.geom_df[self.name_col]
        return out 

        # for i, row in tqdm(df.iterrows()):
        #     name = row[name_col]
        #     shape = row[geom_col]
        #     overlap = self._compute_overlap(shape)
        #     out[name] = overlap 
        # out.index = self.geom_df[self.name_col]
        # return out 

    def _compute_overlap(self, shape):
        eps = 1e-99
        geom = AnyGeom(shape).as_geometry()
        env = geom.GetEnvelope()
        wh = list(self.geom_tree.intersection(env))  

        #frac_overlap = np.zeros(self.size, dtype=float)
        frac_overlap = pd.Series(0, index=self.geom_df.index)
        #wh is a list of index locations, use .loc not .iloc
        for i in wh:
            gi = self.geom_df[self.geom_col].loc[i]
            intersection = geom.Intersection(gi)
            frac_overlap[i] = intersection.Area() / gi.Area() 
        return frac_overlap

    def plot(self, *args, **kwargs):
        for geom in self.geom_df[self.geom_col]:
            fgplots.plot_shape(geom, *args, **kwargs)
