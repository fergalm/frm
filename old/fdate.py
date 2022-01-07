from ipdb import set_trace as idebug
import datetime
import numpy as np
import pytz 
import time 

def strtounix(val, tz='local', fmt='%Y-%m-%d %H:%M:%S%z'):
    
    if len(val) >11:
        fmt = fmt.replace(' ', val[10], 1)

    if len(val) < 20:
        fmt = fmt[:len(val)-2]
    print(fmt)
    
    dt = datetime.datetime.strptime(val, fmt)
    tzinfo = dt.tzinfo 
    if tzinfo is not None and tz != 'local':
        raise ValueError("timezone specified in both string and tz arg")


    if tzinfo is not None:
        tz = tzinfo.tzname(dt)
        tz_offset_sec = tzinfo.utcoffset(dt).total_seconds()
        unixtime = dt.timestamp()
    elif tz == 'local':
        tz_offset_sec = time.timezone
        unixtime = dt.timestamp()
    else:

        tzinfo = pytz.timezone(tz) 
        tz_offset_sec = tzinfo.utcoffset(dt).total_seconds()

        #dt is in naive time. When I try to convert to unixtime,
        #it will be treated as local time. So I must correct to uxt
        unixtime = dt.timestamp() - time.timezone

    return unixtime, tz, tz_offset_sec


