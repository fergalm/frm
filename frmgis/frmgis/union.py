
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
    assert len(geom_list) > 0 
    
    if len(geom_list) == 1:
        return geom_list[0]
    
    if len(geom_list) == 2:
        return geom_list[0].Union(geom_list[1])
        

    size = int(np.floor(len(geom_list)/2))
    geom1 = union_collection(geom_list[:size])
    geom2 = union_collection(geom_list[size:])
    return geom1.Union(geom2)




# def is_clockwise(shape):
#   """Not working yet"""
#     gtype, arr = AnyGeom(shape).as_array()
#     if gtype.lower() != 'polygon':
#         raise ValueError("Not a polygon")
    
#     #Translate to origin
#     arr = arr.copy()
#     arr[:,0] -= np.mean(arr[:,0])
#     arr[:,1] -= np.mean(arr[:,1])
#     dx = np.diff(arr[:,0])
#     dy = np.diff(arr[:,1])

#     sgn = np.sign(np.arctan2(dy, dx))
#     return - np.sum(sgn)

