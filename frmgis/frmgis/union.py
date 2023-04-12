
import numpy as np 


def union_collection(geom_list):
    """Rapidly combine a list of geometries into one.
    
    gdal provides a union-collection function, but it isn't callable
    from Python. This function reproduces the algorithm
    
    Recursively splits the input list in two until it has either one
    or two elements. One element is returned, two are merged. 
    
    This is much faster than combining one at a time because the time
    to union two geometries increases with size. This approach reduces
    the number merges of large objects.
    
    Inputs
    ---------
    geom_list
        A list of gdal/ogr geometries
        
    Returns
    -----------
    A single geometry representing the union of the inputs
    """
    
    if len(geom_list) == 1:
        return geom_list[0]
    
    if len(geom_list) == 2:
        return geom_list[0].Union(geom_list[1])
        
    size = int(np.floor(len(geom_list)/2))
    geom1 = union_collection(geom_list[:size])
    geom2 = union_collection(geom_list[size:])
    return geom1.Union(geom2)

