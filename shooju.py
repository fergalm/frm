import pandas as pd
import shooju
import os

from  ipdb import set_trace as idebug 

"""
Playing at importing data from Shooju
"""


def connect_to_shooju(api_key, user='fergal.mullally', server='http://shooju.exelonds.com'):
    """Create a client to connect to Shooju

    Inputs
    ----------
    api_key
        (str) Either an api key string, or a path to a file containing that key

    Returns
    ---------
    A database connection object. Pass to query_to_ts()
    
    You may need to be on VPN to get this to work. Unfortunately, I can't remember
    how I got a Shooju key, but it was possibly through AGS.
    
    TODO: User/pass for getsecret should be passed in as arguments
    """

    if os.path.exists(api_key):
        api_key = load_key(api_key)

    sj = shooju.Connection(server=server, user=user, api_key=api_key)
    return sj 


def query_for_ts(
    sj, 
    tsdbid, 
    df='-7d', 
    dt='0d', 
    fields=None, 
    max_points=-1, 
    localise=True, 
    reporting_date=None
):

    query = "tsdbid=%s" % (tsdbid)   
    if localise:
        query = query + "@localize"

    if reporting_date is not None:
        date = pd.to_datetime(reporting_date)
        try:
            date = date.tz_convert('UTC')
        except TypeError:
            pass
        datestr = date.strftime("%Y-%m-%dT%H:%M:%SZ")
        query = query + "@repdate:%s" %(datestr)
    
    if fields is None:
        fields = [] 

    res = sj.get_df(
        query, 
        fields, 
        df=df,
        dt=dt,
        max_points=max_points)

    if reporting_date is not None:
        res['version'] = reporting_date
    return res



def load_key(fn):
    """Load shooju API key from a file"""
    with open(fn) as fp:
        key = fp.read()[:-1]
        
    return key



def query_ercot_actual_load(sj, date_from, date_to):
    """Query for a single day of actual hourly load values for ERCOT"""
    tsdbid = 'ERCOT_ACT_LOAD_WTH_ZONE_TOTAL'
    df = query_for_ts(sj, tsdbid, df=date_from, dt=date_to)
    return df


def query_ercot_predicted_load(sj, date_from, date_to):
    tsdbid = 'PRT_ERCOT_LD_DAY_FCST'
    df = query_for_ts(sj, tsdbid, df=date_from, dt=date_to, reporting_date=date_from)
    return df



def get_actual():
    sj = connect_to_shooju()
    tsdbid = 'ERCOT_ACT_LOAD_WTH_ZONE_TOTAL'

    df = query_for_ts(sj, tsdbid, df=str('2002-01-01'), dt='+0d')
    # df.to_csv('../data/ercot_load_actual/load_actual.csv')
    return df



if __name__ == "__main__":
    main()