from ipdb import set_trace as idebug
import numpy as np
import requests 

"""
Implement a solution to the travelling salesman problem
"""

from abc import ABC, abstractmethod
class AbstractDistanceMetric(ABC):
    def __init__(self, cities):
        self.cities = cities 

    @abstractmethod 
    def getTotalDistance(self, route):
        pass 



# # Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
# def compute_path_distance(r, cities):
#     """
#     Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.

#     Private function of `solve_travelling_salesman`
    
#     """
#     return np.sum([np.linalg.norm(cities[r[p]]-cities[r[p-1]]) for p in range(len(r))])



def default_callback(cities, route):
    """Don't do anything"""
    return 


def solve_travelling_salesman(
        cities: np.ndarray, 
        closed=False,
        metric:AbstractDistanceMetric=None,
        callback=default_callback, 
        threshold=1e-6, 
    ):
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

    improvement_factor = 1 
    if closed:
        cities = np.append(cities, cities[:1], axis=0)
        route = np.arange(cities.shape[0]) 
        route[-1] = 0
        #If it's a closed loop, last city immutable
        sentinel = len(route) -1 
    else:
      route = np.arange(cities.shape[0]) 
      sentinel = len(route)

    # route = np.arange(cities.shape[0]) 

    # sentinel = len(route)
    # if closed:
    #     sentinel -= 1  

    metric = metric or EuclideanNorm(cities)
    best_distance = metric.getTotalDistance(route)
    assert np.isfinite(best_distance)
    
    while improvement_factor > threshold: 
        distance_to_beat = best_distance 
        for swap_first in range(1,len(route)-2): 
            for swap_last in range(swap_first+1, sentinel): 
            # for swap_last in range(swap_first+1,len(route)): 
                new_route = two_opt_swap(route,swap_first,swap_last) 
                new_distance = metric.getTotalDistance(new_route)
                if new_distance < best_distance: 
                    route = new_route 
                    best_distance = new_distance 
                    callback(cities, route)
        improvement_factor = 1 - best_distance/distance_to_beat 
    
    if closed:
        #Drop the last location to ensure len(route) == len(cities)
        route = route[:-1]  
    return route 



def example():
    lng = np.random.uniform(-76, -75, 30)
    lat = np.random.uniform(33, 34, 30)
    cities = np.vstack([lng, lat]).transpose()
    cities = np.append(cities, cities[:1], axis=0)

    route = solve_travelling_salesman(cities)
    return route 



def two_opt_swap(route, i, k):
    """
    Reverse the order of all elements from element i to element k in array `route`.

    Private function of `solve_travelling_salesman`
    """
    first = route[0:i]
    middle = route[k:-len(route)+i-1:-1]
    last = route[k+1:len(route)]
    return np.concatenate((first, middle, last))

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



class EuclideanNorm(AbstractDistanceMetric):
    def getTotalDistance(self, route):
        cities = self.cities 
        r = route  #Mnuemonic
        return np.sum([np.linalg.norm(cities[r[p]]-cities[r[p-1]]) for p in range(len(r))])


class GreatCircle(ABC):
    ...

class GoogleMapsDistance(ABC):
    """
    Compute the distance between locations in terms
    of number of seconds of driving time between the.

    See
    https://developers.google.com/maps/documentation/distance-matrix/distance-matrix

    Note
    ---------
    This API is quick and dirty, and doesn't check
    error codes. If something goes wrong, check
    the resp object for details.

    """
    def __init__(self, cities, api_key=None):
        self.api_key = api_key or "AIzaSyC-gsLNcbRQemORIrSipT4yheac0rrizUw"
        self.url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        self.maxCitiesPerQuery = 10
        self.matrix = self.setup(cities)

    def setup(self, cities):
        chunks = chunk(cities, self.maxCitiesPerQuery)
        numChunk = len(chunks)

        #For each pair of origins and destinations, make a query
        collection = np.empty((numChunk, numChunk), dtype=object)
        for i in range(numChunk):
            for j in range(i, numChunk):
                collection[i, j] = self.query(chunks[i], chunks[j])

                #Matrix is roughly symmetrical
                if i != j:
                    collection[j,i] = collection[i,j].transpose()

        #Concatenate the queries 
        rows = []
        for i in range(numChunk):
            rows.append( np.hstack(collection[i,:]))
        return np.vstack(rows)
        
    def query(self, origins, destinations):
            assert len(origins) <= 10
            assert len(destinations) <= 10

            origins = map(getLatLng, origins)
            origins = "|".join(origins)

            destinations = map(getLatLng, destinations)
            destinations = "|".join(destinations)

            params = {
                'departure_time': 'now',
                'destinations': destinations,
                'origins': origins,
                'key': self.api_key
            }

            result = self.makeRequest(params)
            return self.parse(result)

    def makeRequest(self, params) -> dict:
            """requests call pulled out for easier mocking"""
            print("Making request")
            resp = requests.get(self.url, params=params)
            resp.raise_for_status()
            return resp.json()
    
    def getTotalDistance(self, route):
        if self.matrix is None:
            raise RuntimeError("Call setup before getTotalDistance")
        
        #TODO. Vectorize this
        dist = 0
        for r in range(len(route)-1):
            i = route[r]
            j = route[r+1]
            dist += self.matrix[i, j]
        return dist 


    def parse(self, data):
        if data['status'] != "OK":
            raise IOError(f"Google Distance API request failed with status {data['status']}")

        data = data['rows']
        nrows = len(data)
        row0 = data[0]
        ncols = len(row0['elements'])
        out = np.zeros((nrows, ncols))

        for i, row in enumerate(data):
            elts = row['elements']
            out[i] = list(map(lambda x: x['duration']['value'], elts))

        duration_sec = out 
        return duration_sec


def getLatLng(row):
    lng, lat = row 
    return f"{lat:.7f},{lng:.7f}"



def chunk(arr, chunkSize):
    limits = np.arange(0, len(arr), chunkSize)
    limits = np.append(limits, len(arr))

    chunks = []
    for i in range(len(limits)-1):
        lwr, upr = limits[i], limits[i+1]
        chunks.append( arr[lwr:upr] )
    return chunks


    # x, y = np.meshgrid(list1[:,0], list2[:,0])
    # return x+y 


def test_foo():

    cities = np.array([
       [-76.61868812,  39.41444055],
       [-76.78895515,  39.38911359],
       [-76.41321639,  39.40305095],
       [-76.8213196 ,  39.3924616 ],
       [-76.41406096,  39.40232465],
       [-76.74915911,  39.27813292],
       [-76.59255322,  39.39573046],
       [-76.52821884,  39.40940237],
       [-76.57444495,  39.6967646 ],
       [-76.74038349,  39.27907313],
       [-76.74048349,  39.17907313],
    ])
    size = len(cities)

    foo = GoogleMapsDistance(cities)
    assert foo.matrix.shape == (size, size)
    assert foo.matrix.trace() == 0

    def debug(cities, route):
        idebug()

    solve_travelling_salesman(cities, metric=foo, callback=debug)
    return foo