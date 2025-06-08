# -*- coding: utf-8 -*-

from ipdb import set_trace as idebug
from pprint import pprint
import osgeo.osr as osr
import pandas as pd
import numpy as np
import requests
import os


import frmgis.get_geom as get_geom

"""
Tools to query the US Census

Built off of api.census.gov

Documentation
---------------------
See https://www.census.gov/data/developers/guidance/api-user-guide.html


Overview
-----------------
There are two classes in this package

CensusQuery()
    Query numbers for geographies

TigerQuery()
    Query geometries for geographies


The available census information is overwhelming. This API only accesses
a slice of it. To use it effectively, there are a couple of concepts
you need to understand.

Surveys
-------
The census is best known for the "Decennial Survey" that tries to count
everyone once every decade. But smaller surveys are released every year.
The most common of these is the American Communities Survey, which
is released in 1, 3 and 5 year increments. The smaller the number of
years, the more narrow and deep the survey is.

The surveys are organised on-line by year, survey type, and table
Survey type can be dec (for the 10 year survey, or acs1, acs3, acs5).
The full list of surveys available for a given year is available in
json at

    https://api.census.gov/data/YYYY/  (e.g YYYY=2015)

Within each survey are 30 or more tables, in which the data are stored.

Some popular tables include

Decennial
    dec/sf1
        (Population and race statistics)
    dec/sf2
        (Aggregated answers to questions people answered in the survey
    acs/acs1
        ACS 1 year survey
    acs/acs3
        ACS 3 year survey
    acs/acs4
        ACS 5 year survey

FIPS
----
Census data is aggregated in a hierarchical format of
State - County - Tract - Block Group - Block.

There are other formats, but we ignore them in this module.
A block is often too small for the census to release much information
(for privacy reasons), and most work happens at the block-group level.
A block group has a population of 600-2000. A tract is somewhat larger.

Each block is has a unique identifer, known as a FIPS code. The
FIPS code is built up as  01-234-56789-0-123

01
    State number. See state_to_fips_code
234
    County number
56789
    Tract
0
    Block Group
123
    Block

This module allows you to query by county, tract, or block group.
A fips code of any resolution is accepted. For example, querying
for county level info of 01-234-56789-0-123 will return information
for 01-234. Querying For tract information for 01-234 will return
information for all tracts in that county.


Columns to query
------
There are hundreds of features collected by the census for
each block group. You can find the complete list for a table
at, for example, https://api.census.gov/data/2010/dec/sf1/variables.json

The total population column is called P001001 in the decennial census,
and B01001_001Ein ACS5 (note: B00001_001E indicated the actual
number of people sampled, which will be much less)


Geometries
------------
Geometries are stored in the Census' TIGER system, and are aggregated
at the state level. This module exposes an interface to query for
tracts and block groups.


Example Usage
----------------
::
    tq = census.TigerQuery()
    tq.query_tract(2015, '24005')  #Baltimore County, MD

    c = Census()
    year = 2010
    src = 'dec'
    table = 'sf1'
    cols = ['P001001']

    # Query a single block group
    fips = '250054007011'
    c.query_block_group(year, src, table fips, cols )

    # Query every block group in a tract
    fips = '25005400701'  #Note, one character shorter
    c.query_block_group(year, src, table fips, cols )

    # Query every block group in a county
    fips = '25005'
    c.query_block_group(year, src, table fips, cols )

    #Queries a single county, 25-005
    fips = '250054007011'
    c.query_county(year, src, table, fips, cols )

    #Query ACS5
    c.query_block_group(2021, 'acs', 'acs5',  24005, cols)
A similar syntax applies for querying for tracts, counties and states.
The server may set limits on how many levels you can query simultaneously.
For example, querying every block group in a state isn't allowed in a
single query.

"""

DEFAULT_KEY = '40c7ae9821edae671b516c3249d4415a00ab3805'




class CensusQuery():
    """
    Query tablular data from the Census. See module level docs.

    Each method takes the following arguments/

    year
        (int or str) Year of survey to query
    src
        (str) e.g dec or acs
    table
        (str) e.g sf1, or acs1
    fips
        (int or str) FIPs code to search.
    cols
        (list of strings) Which columns to request.
    """

    def __init__(self, api_key=None):
        self.base_url = "https://api.census.gov/data"
        self.api_key = api_key

    def query_county(self, year, src, table, fips, cols):
        tokens = explode_fips(fips)  #state, county, [tract, [bg]]
        tokens = self.add_wildcards(tokens, 2)
        state, county = tokens[:2]

        predicates = dict()
        predicates['get'] =",".join(cols)
        predicates['for'] = 'county:%s' %(county)
        predicates['in'] = ['state:%s' %(state)]
        return self.query(year, src, table, predicates)

    def query_tract(self, year, src, table, fips, cols):
        tokens = explode_fips(fips)  #state, county, [tract, [bg]]
        tokens = self.add_wildcards(tokens, 3)
        state, county, tract = tokens[:3]

        predicates = dict()
        predicates['get'] =",".join(cols)
        predicates['for'] = 'tract:%s' %(tract)
        predicates['in'] = ['state:%s' %(state),
                            'county :%s' %(county)]
        return self.query(year, src, table,predicates)

    def query_block_group(self, year, src, table, fips, cols):
        tokens = explode_fips(fips)  #state, county, [tract, [bg]]
        tokens = self.add_wildcards(tokens, 4)

        state, county, tract, block_group = tokens[:4]
        predicates = dict()
        predicates['get'] =",".join(cols)
        predicates['for'] = 'block group:%s' %(block_group)
        predicates['in'] = ['state:%s' %(state),
                            'county:%s' %(county),
                            'tract:%s' %(tract)]

        pprint(locals())
        return self.query(year, src, table, predicates)

    def query_block(self, year, src, table, fips, cols):
        """This only works for decennial census?"""
        tokens = explode_fips(fips)  #state, county, [tract, [bg]]
        tokens = self.add_wildcards(tokens, 5)

        state, county, tract, block_group, block = tokens[:5]
        predicates = dict()
        predicates['get'] =",".join(cols)
        predicates['for'] = 'block:%s' %(block)
        predicates['in'] = ['state:%s' %(state),
                            'county:%s' %(county),
                            'tract:%s' %(tract),
                            # 'block group:%s' %(tract)
                            ]

        return self.query(year, src, table, predicates)

    def add_wildcards(self, tokens, n_elt):
        """Adds wild cards to the list of fips tokens until the length
        is at least n_elt"""
        while len(tokens) < n_elt:
            tokens.append('*')
        return tokens

    def query(self, year, src, table, predicates):
        url = "/".join([self.base_url, str(year), src, table])

        if self.api_key is not None:
            predicates['key'] = self.api_key

        r = requests.get(url, params=predicates)
        if not r.ok:
            raise ValueError("Error code %i: %s\n%s" %(r.status_code, r.text, r.url, ))

        if len(r.text) == 0:
            cols = predicates['get'].split(',')
            return pd.DataFrame(columns=cols)  #Empty dataframe

        jstext = r.json()
        df = pd.DataFrame(columns=jstext[0], data=jstext[1:])
        # #Convert cols to ints as appropriate.
        # #TODO Do I ever need floats?
        # for col in df.columns:
        #     try:
        #         df[col] = df[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass

        df['fips'] = self.make_fips_from_result(df) #String, not an int
        return df

    def make_fips_from_result(self, df):
        state = df.state

        cols = df.columns
        if ('block' in cols) and ('block group' not in cols):
            df['block group'] = df.block.str[0]
            df['block'] = df.block.str[1:]

        args = [state]
        # county, tract, block_group, block = None, None, None, None
        for c in ["county", "tract", "block group", "block"]:
            if c in df.columns:
                args.append(df[c])
            else:
                return make_many_fips(*args)
        return make_many_fips(*args)


class TigerQuery():
    """Query for Census Geometries. See module level docs for more information"""

    def __init__(self, cache="./"):
        """
        Inputs
        -----
        cache
            (str) Location to cache zip files downloaded from TIGER
        """
        self.url = "http://www2.census.gov/geo/tiger/"
        self.cache_path = cache
        #For some reason, the name of the FIPS columns in the
        #shapefiles changes between versions
        self.fips_alias_in_shpfile = "NOT DEFINED"
        
        if not os.path.exists(cache):
            try:
                os.mkdir(cache)
            except OSError as e:
                raise OSError(f"Cache directory {cache} does not exist and couldn't be created")

    def query_county(self, year, fips):
        """Query for all tracts in a state for a given year

        Inputs
        ----------
        year
            (int or string). Not all years have data available.
        fips
            (int or str) FIPS code of the tract to query. Only the first two
            digits (the state id) are used, the rest is ignored

        Returns
        ---------
        A dataframe with columns of NAME and geom. geom contains
        geometry objects, not WKTs.
        """
        state = explode_fips(fips)[0]
        ftype, year, state = self.santize_inputs('county', year, state)
        df =  self.query(ftype, year, state)
        return df[df.STATEFP == state].copy()
            

    def query_congress(self, year, fips):
        """
        Before 2022, congressional districts are stored in a single file.
        AFter 2022 they are stored one file per state.
        
        I want to abstract over this difference. 
        
        This is a draft of such a code, but it needs to be cleaned
        up some, I think
        

        """
        state = explode_fips(fips)[0]
        
        #2020 is the 117th congress
        congress = int(np.floor(year/2) - 893)
        ftype = f"cd{congress}"
        fn = self.get_filename(ftype, year, "us")

        df =  self.query(ftype, year, state)
        try:
            return df[df.STATEFP == state].copy()
        except KeyError:
            return df[df.STATEFP20 == state].copy()
        
    def query_tract(self, year, fips):
        """Query for all tracts in a state for a given year

        Inputs
        ----------
        year
            (int or string). Not all years have data available.
        fips
            (int or str) FIPS code of the tract to query. Only the first two
            digits (the state id) are used, the rest is ignored

        Returns
        ---------
        A dataframe with columns of NAME and geom. geom contains
        geometry objects, not WKTs.
        """
        state = explode_fips(fips)[0]
        ftype, year, state = self.santize_inputs('tract', year, state)
        return self.query(ftype, year, state)

    def query_block_group(self, year, fips):
        """Query for all block groups in a state for a given year

        UNTESTED

        Inputs
        ----------
        year
            (int or string). Not all years have data available.
        fips
            (int or str) FIPS code of the tract to query. Only the first two
            digits (the state id) are used, the rest is ignored

        Returns
        ---------
        A dataframe with columns of NAME and geom. geom contains
        geometry objects, not WKTs.
        """
        state = explode_fips(fips)[0]
        ftype, year, state = self.santize_inputs('bg', year, state)
        return self.query(ftype, year, state)

    def query_block(self, year, fips):
        """Query for all block groups in a state for a given year

        UNTESTED

        Inputs
        ----------
        year
            (int or string). Not all years have data available.
        fips
            (int or str) FIPS code of the tract to query. Only the first two
            digits (the state id) are used, the rest is ignored

        Returns
        ---------
        A dataframe with columns of NAME and geom. geom contains
        geometry objects, not WKTs.
        """
        state, county = explode_fips(fips)[:2]
        ftype, year, state = self.santize_inputs('tabblock', year, state)

        # import ipdb; ipdb.set_trace()
        if int(year) < 2020:
            assert county is not None
            df = self.query(ftype, year, state, county)
        else:
            two_digit_year = str(year)[-2:]
            # two_digit_year = 10
            fn = f"tl_{year}_{state}_tabblock{two_digit_year}.zip"
            df = self.query(ftype, year, state, fn=fn)
        
        df = filter_to_fips(df, year, fips)
        return df
            
    def query(self, ftype, year, state, county=None, fn=None):

        cache_path = self.get_cache_path(ftype, year, state, county)
        if not os.path.exists(cache_path):
            self.download(ftype, year, state, county, fn)
        df = self.convert_zip_to_wgs84_df(cache_path, year)
        return df

    def download(self, ftype, year, state, county=None, fn=None):
        fn = fn or self.get_filename(ftype, year, state, county)
        url = self.make_url(year, ftype, county, fn)
        print(url)

        cache_file = self.get_cache_path(ftype, year, state, county)
        r = requests.get(url)
        if not r.ok:
            raise IOError("Error %i for url %s" %(r.status_code, url))

        with open(cache_file, 'wb') as fp:
            fp.write(r.content)
        return cache_file

    def make_url(self, year, ftype, county, fn):
        if ftype[:2].upper() == 'CD':
            url = os.path.join(self.url,
                            f"TIGER{year}",
                            f"CD",
                            fn)
        elif county is None:
            if int(year) >= 2020 and False:
                two_digit_year = str(year)[-2:]
                url = os.path.join(self.url,
                                f"TIGER{year}",
                                f"{ftype.upper()}{two_digit_year}" ,
                                fn)
            else:
                url = os.path.join(self.url,
                                f"TIGER{year}",
                                f"{ftype.upper()}",
                                fn)
        return url 
    

    def convert_zip_to_wgs84_df(self, zip_file, year):

        fips_alias = self.get_fips_alias_in_shapefile(year)

        try:
            df = get_geom.load_geoms_as_df(zip_file, "GEOID")
            key = "GEOID"
        except KeyError:
            try:
                df = get_geom.load_geoms_as_df(zip_file, "GEOID20")
                key = "GEOID20"
            except KeyError:
                df = get_geom.load_geoms_as_df(zip_file, "GEOID10")
                key = "GEOID10"

        df = df.rename({key: 'GEOID'}, axis=1)

        source = osr.SpatialReference()
        source.ImportFromEPSG(4269)  #NADS83
        dest = osr.SpatialReference()
        dest.ImportFromEPSG(4326)  #WGS84
        transform = osr.CoordinateTransformation(source, dest)

        #Note, this used to be geoms, and I changed it to geom
        #if geom fails, put in better logic
        for i in range(len(df)):
            df.geom.iloc[i].Transform(transform)
        return df


    def santize_inputs(self, ftype, year, state, county=None):
        year = int(year)
        assert year > 1989
        year = str(year)

        state = int(state)  #Guard against MD, TX, etc
        state = "%02i" %(state)

        if county is not None:
            county = int(county)
            county = "%03i" %(county)

        ftype = ftype.upper()
        return ftype, year, state

    def get_cache_path(self, ftype, year, state, county=None):
        fn = self.get_filename(ftype, year, state, county)
        return os.path.join(self.cache_path, fn)

    def get_filename(self, ftype, year, state, county):
        raise NotImplementedError

    def get_fips_alias_in_shapefile(self, year):
        """Get the name of the column in the shapefile that includes the fips id

        The decennial census uses a different name for each decade, while
        ACS does not. The ACS class overrides this method"""
        #return "GEOID%s" %(str(year)[-2:])
        return "GEOID"


class TigerQueryAcs(TigerQuery):
    def __init__(self, cache_path):
        TigerQuery.__init__(self, cache_path)
        # self.fips_alias_in_shpfile = "GEOID"

	#tl_2020_us_cd116.zip	
    def get_filename(self, ftype, year, state, county=None):
        if ftype.upper() == "COUNTY":
            return f"tl_{year}_us_county.zip"
        
        fn = f"tl_{year}_{state}_{ftype.lower()}.zip"
        return fn

    def get_fips_alias_in_shapefile(self, year):
        """The ACS doesn't use the year in the fips alias column"""
        return "GEOID"


def filter_to_fips(df: pd.DataFrame, year:int, fips:int) -> pd.DataFrame:
    fips_values = explode_fips(fips)
    fips_cols = get_tiger_col_name_for_fips(year)
    
    idx = np.ones(len(df), dtype=bool)
    for i in range(len(fips_values)):
        col = fips_cols[i]
        idx &=  df[col] == fips_values[i] 

    return df[idx].copy()


def get_tiger_col_name_for_fips(year:int):
    """TODO IF the year is 2021, be smart enough to fall back to 2020"""
    opts = {
        2020: 'STATEFP20 COUNTYFP20 TRACTCE20 BLOCKCE20'.split()
    }
    
    year = int(year)
    return opts[year]



class TigerQueryDec(TigerQuery):
    def __init__(self, cache_path):
        TigerQuery.__init__(self, cache_path)

    def get_filename(self, ftype, year, state, county):
        year2 = str(year)[-2:]  #two digit year
        ftl = ftype.lower()

        if county is None:
            # fn = f"tl_{year}_{state}_{ftl}{year2}.zip"
            fn = f"tl_{year}_{state}_{ftl}.zip"
        else:
            #fn = f"tl_{year}_{state}{county}_{ftl}{year2}.zip"
            fn = f"tl_{year}_{state}{county}_{ftl}.zip"
        return fn


def make_many_fips(state, county=None, tract=None, block_group=None, block=None):
    """Wrapper around make_fips to deal with lists

    Inputs
    -------
    state, county, tract, block_group, block
        1d iterables or None

    Returns
    ---------
    A list of fips codes


    Example
    ---------
    state = [24, 24, 24]
    county = [3, 4, 5]

    make_many_fips(state, county)
    >>> ['25003', '24004', '24005']
    """

    args = []
    for arg in [state, county, tract, block_group, block]:
        if arg is not None:
            args.append(arg)

    fips = map(make_fips, *args)
    return list(fips)


def make_fips(state, county=None, tract=None, block_group=None, block=None):
    fips = ["%02i" %(int(state))]

    if county is not None:
        fips.extend("%03i" %(int(county)))

    if tract is not None:
        #tract might be given as 1234.01 or 123401
        tract = float(tract)
        if tract > int(tract):
            tract *= 100  #Remove the decimal place

        fips.extend("%06.0f" %(tract))

    if block_group is not None:
        fips.extend("%1i" %(int(block_group)))

    if block is not None:
        fips.extend("%03i" %(int(block)))

    return "".join(fips)


def explode_fips(fips):
    """Convert a FIPS to a list of [state, county, block group, block]

    For incomplete inputs, a truncated list is returned. For example
    24005 returns [state, county]
    """

    fips = str(fips)
    if len(fips) < 2:
        raise ValueError("fips must include at least a state")

    state = fips[:2]

    if len(fips) <= 2:
        return [state]
    county = fips[2:5]

    if len(fips) <= 10:
        return [state, county]
    tract = fips[5:11]

    if len(fips) <= 11:
        return [state, county, tract]
    block_group = fips[11]

    if len(fips) <= 12:
        return [state, county, tract, block_group]

    block = fips[12:]
    return [state, county, tract, block_group, block]



def state_to_fips_code(state_name):
    """Convert a state abbreviation to a fips code."""
    states = dict(
        AL = '01',       AK ='02',       AZ = '04',
        AR = '05',       CA ='06',       CO = '08',
        CT = '09',       DE ='10',       DC = '11',
        FL = '12',       GA ='13',       HI = '15',
        ID = '16',       IL ='17',       IN = '18',
        IA = '19',       KS ='20',       KY = '21',
        LA = '22',       ME ='23',       MD = '24',
        MA = '25',       MI ='26',       MN = '27',
        MS = '28',       MO ='29',       MT = '30',
        NE = '31',       NV ='32',       NH = '33',
        NJ = '34',       NM ='35',       NY = '36',
        NC = '37',       ND ='38',       OH = '39',
        OK = '40',       OR ='41',       PA = '42',
        RI = '44',       SC ='45',       SD = '46',
        TN = '47',       TX ='48',       UT = '49',
        VT = '50',       VA ='51',       WA = '53',
        WV = '54',       WI ='55',       WY = '56',
        )

    return states[state_name]


import pytest
def test_make_fips():
    fips = make_fips(24)
    assert fips == "24"

    fips = make_fips(24, 5)
    assert fips == "24005"

    fips = make_fips(24, "5")
    assert fips == "24005"

    with pytest.raises(ValueError):
        fips = make_fips(24, "5", 4001)

    fips = make_fips(24, "5", 4001.01)
    assert fips == "24005400101", fips

    fips = make_fips(24, "5", 400101)
    assert fips == "24005400101", fips

