from ipdb import set_trace as idebug
import numpy as np

"""
Implement a solution to the travelling salesman problem
"""

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
def compute_path_distance(r, c):
    """
    Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.

    Private function of `solve_travelling_salesman`
    
    """
    return np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])


def two_opt_swap(route, i, k):
    """
    Reverse the order of all elements from element i to element k in array `route`.

    Private function of `solve_travelling_salesman`
    """
    first = route[0:i]
    middle = route[k:-len(route)+i-1:-1]
    last = route[k+1:len(route)]
    return np.concatenate((first, middle, last))



def default_callback(cities, route):
    """Don't do anything"""
    return 


def plotting_callack(cities: np.ndarray, route: np.ndarray):
    import matplotlib.pyplot as plt 
    plt.clf()
    for i in range(len(route)-1):
        start = route[i]
        end = route[i+1]

        lng1, lat1 = cities[start]
        lng2, lat2 = cities[end]
        #I can be faster than this
        plt.plot( [lng1, lng2], [lat1, lat2], 'ko-')
    plt.pause(.5)


def solve_travelling_salesman(cities: np.ndarray, threshold=.001, callback=default_callback, closed=False):
    """
    Iteratively solve the travelling salesman problem.

    The travelling salesman is the problem of finding the
    shortest route between a collection of points. 

    This implmentation cribbed from 
    https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy

    and is based on the discussion on Wikipedia at 
    https://en.wikipedia.org/wiki/2-opt

    This particular implementation is for points on a Euclidean space.
    For lng/lat positions, scale the longitude points by dividing
    by the cosine of longitude for a more accurate result. 

    Inputs
    ------------
    cities  
        A collection of x,y points. Each row represents a destination
        on the itinary
    theshold
        Continue iterating until no improvement yields a fractional reduction
        in distance less than this value 
    callback
        A function called every time an improved path is found 
        Signature is `callback(cities, route)`
    closed
        If **True**, the first point in cities is appended to the
        array so the path starts and ends at the same location


    Returns 
    ------------
    route
        A 1d np.array of indices into `cities`. `cities[route]`
        is the shortest path found.
    """
    assert cities.ndim == 2 
    assert cities.shape[1] == 2

    route = np.arange(cities.shape[0]) 
    improvement_factor = 1 

    best_distance = compute_path_distance(route,cities) 
    while improvement_factor > threshold: 
        distance_to_beat = best_distance 
        for swap_first in range(1,len(route)-2): 
            for swap_last in range(swap_first+1,len(route)): 
                new_route = two_opt_swap(route,swap_first,swap_last) 
                new_distance = compute_path_distance(new_route,cities) 
                if new_distance < best_distance: 
                    route = new_route 
                    best_distance = new_distance 
                    default_callback(cities, route)
        improvement_factor = 1 - best_distance/distance_to_beat 
    return route 



def example():
    lng = np.random.uniform(-76, -75, 30)
    lat = np.random.uniform(33, 34, 30)
    cities = np.vstack([lng, lat]).transpose()
    cities = np.append(cities, cities[:1], axis=0)

    route = solve_travelling_salesman(cities)
    return route 