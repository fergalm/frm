from ipdb import set_trace as idebug
from pprint import pprint
import osgeo.osr as osr
import pandas as pd
import numpy as np
import requests
import os

from typing import List 

from frmbase.support import npregex, lmap
from .abstractgeocoder import AbstractGeocoder

DEFAULT_KEY="AIzaSyC-gsLNcbRQemORIrSipT4yheac0rrizUw"

class GoogleGeoCoder(AbstractGeocoder):
    """
    https://developers.google.com/maps/documentation/geocoding/start
    """
    def __init__(self, api_key=DEFAULT_KEY):
        self.api_key = api_key
        self.url = 'https://maps.googleapis.com/maps/api/geocode/json'
    
    def query(self, address):
        params = {
            'address': address,
            'key': self.api_key,
        }

        resp = requests.get(self.url, params=params)
        resp.raise_for_status()
        result = resp.json()['results']
        result = self.parse(result)
        return result

    def parse(self, result:List[dict]):
        out = lmap(self.parseSingleMatch, result)
        return out


    def parseSingleMatch(self, match:dict) -> dict :      

        out = {
            'fullAddress': match['formatted_address'],
        }

        coords = match['geometry']['location']
        out['lng_deg'] =  coords['lng']
        out['lat_deg']=  coords['lat']
        return out 



