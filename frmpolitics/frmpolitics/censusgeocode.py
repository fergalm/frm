from ipdb import set_trace as idebug
from pprint import pprint
import numpy as np
import requests
import re 

from frmbase.support import npregex, lmap
from .abstractgeocoder import AbstractGeocoder


class CensusGeoCoder(AbstractGeocoder):
    def __init__(self):
        self.url = url = "https://geocoding.geo.census.gov/geocoder/geographies/"

    def query(self, address: str, parse=True) -> dict:
        """
        https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html

        """
        url = f"{self.url}/onelineaddress"
        
        params = {
            'address': address,
            'benchmark' :'Public_AR_Current',
            'vintage' :'Current_Current',
            'format': 'json',
        }    

        resp = requests.get(url, params=params)
        resp.raise_for_status()

        result = resp.json()['result']
        if parse:
            result = self.parse(result)
        return result

    def reverse_query(self, lng_deg, lat_deg):
        url = f"{self.url}/coordinates"
        params = {
            'x': lng_deg,
            'y': lat_deg,
            'benchmark' :'Public_AR_Current',
            'vintage' :'Current_Current',
            'format': 'json',
        }    
        resp = requests.get(url, params=params)
        resp.raise_for_status()

        result = resp.json()['result']
        return self.parseReverse(result)

    def parseReverse(self, result):
        out = {
            'lng_deg': result['input']['location']['x'],
            'lat_deg': result['input']['location']['y'],
        }

        #Extract out district information
        geographies = result['geographies']
        out.update( parseGeographies(geographies) )
        return out


    def parse(self, result):
        """This assumes only one address gets returned"""
        matches = result['addressMatches']
                    
        out = lmap(self.parseSingleMatch, matches)
        return out


    def parseSingleMatch(self, match:dict) -> dict :      
        out = {
            'fullAddress': match['matchedAddress'],
            'state': match['addressComponents']['state'],
            'zip': match['addressComponents']['zip'],
            'city': match['addressComponents']['city'],
            'lng_deg': match['coordinates']['x'],
            'lat_deg': match['coordinates']['y'],
            'tigerLineId': match['tigerLine']['tigerLineId'],
            'tigerSide': match['tigerLine']['side'],
        }

        #Extract out district information
        geographies = match['geographies']
        out.update( parseGeographies(geographies) )
        # keys = list(geographies.keys())
        
        # #Get FIPs
        # wh = np.where(npregex("Census Blocks", keys))[0][0]
        # key = keys[wh]
        # out['fips'] = geographies[key][0]['GEOID']

        # #Get Congressional. Is returned as, e.g 2402
        # #First two digits are state id, second are district id
        # # idebug()
        # wh = np.where(npregex("Congressional", keys))[0][0]
        # key = keys[wh]
        # out['Congress'] = geographies[key][0]['GEOID']

        # #Get Statehouse. May only work in Maryland
        # wh = np.where(npregex("Legislative Districts - Lower", keys))[0][0]
        # key = keys[wh]
        # out['Leg'] = geographies[key][0]['BASENAME']

        # #Get County. May only work in Maryland
        # wh = np.where(npregex("County", keys))[0][0]
        # key = keys[wh]
        # out['County'] = geographies[key][0]['BASENAME']

        return out 


def parseGeographies(geographies: dict):
    out = {}
    keys = list(geographies.keys())
    
    #Get FIPs
    wh = np.where(npregex("Census Blocks", keys))[0][0]
    key = keys[wh]
    out['fips'] = geographies[key][0]['GEOID']

    #Get Congressional. Is returned as, e.g 2402
    #First two digits are state id, second are district id
    # idebug()
    wh = np.where(npregex("Congressional", keys))[0][0]
    key = keys[wh]
    out['Congress'] = geographies[key][0]['GEOID']

    #Get Statehouse. May only work in Maryland
    wh = np.where(npregex("Legislative Districts - Lower", keys))[0][0]
    key = keys[wh]
    out['Leg'] = geographies[key][0]['BASENAME']

    #Get County. May only work in Maryland
    wh = np.where(npregex("Counties", keys))[0][0]
    key = keys[wh]
    county = geographies[key][0]['NAME']
    county = re.sub("[ ',\.]", "", county)
    out['County'] = county

    return out 
