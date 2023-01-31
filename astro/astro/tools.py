# -*- coding: utf-8 -*-
import re
import math
import numpy as np
import datetime
import inspect


def jrdToBkjd(val):
    return val+67.0

def bkjdToMbjd(val):
    return val+54832.5

def mbjdToBkjd(val):
    return val-54832.5

def rbjdToBkjd(val):
    return toFloat(val)-54833.0


def bjdToBkjd(val):
    return val - 2454833.0

def fixKoiString(val):
    if re.match("^K\d\d\d\d\d.\d\d", val):
        #Already in the right format
        return val

    num = toFloat(val)
    if num==0:
        raise ValueError("Can't parse %s into a KOI" %(str(val)))

    out = "K%08.02f" %(num)
    return re.sub("\.00", ".01", out)


def minorKoi(koiString):
    reObj = re.match("K.....\.(\d\d)", koiString)
    return int(reObj.group(1))

def majorKoi(koiString):
    reObj = re.match("K(.....)\.\d\d", koiString)
    return int(reObj.group(1))


def createTceId(kepid, planetNum):
    """Create a unique number (for a quarter) for a TCE based on its kepid and
    planet number in DV. Useful for comparing lists of TCEs

    Input:
    kepid       (float or 1d array)
    planetNum   (float or 1d array)

    Return
    float or 1d array depending on input
    """

    tceId = 100*(kepid + planetNum/100.)

    #If inputs are numpy, we can cast to an int
    if isinstance(tceId, np.ndarray):
        tceId = np.round(tceId).astype(np.int64)
    return tceId



def toFloat(strr):
    try:
        return float(strr)
    except ValueError:
        return 0
    except TypeError:
        return 0


def isScalar(x):
    """Returns true is x is a single object, but false if it
    is an array.

    If isScalar(x), then len(x) throws an exception
    """

    return not hasattr(x, "__len__")


def ppmToMilliMag(ppm):
    assert(ppm < 1e6)

    val = 1- ppm/1e6
    return -2.5*math.log10(val)

def milliMagToPpm(mmag):
    val = 1 - 10**(mmag/-2500)
    return 1e6*val


def isClose(a,b, tol):
    """Synonym for checkClose"""
    return checkClose(a,b, tol)

def checkClose(a, b, tol=1e-5):
    a = toFloat(a)
    b = toFloat(b)
    if math.fabs(a-b) > tol:
        return False
    return True


def getch():
    raw_input()


def grep(pattern, filename):
    """Simple replacement for a command line grep.

    Sometimes you want to check for the existence of something in a
    file without going through the hassle of reading it in. This
    is a simple replacement for the grep tool. It doesn't have the
    power of it's command line brother, but it's often enough.
    """

    out = []

    fp = open(filename)
    text = fp.readlines()
    fp.close()

    expr = re.compile(pattern)
    for line in text:
        if re.search(expr, line):
            out.append(line)
    return out



def printse(val, err, sigdig=2, pretty=True):
    """Return a value and its uncertainty in SI format, and
    scientific notation, eg 12.345(67)e7

    Input
    val, err (float) The value and it's associated uncertainty
    sigdig (int)     Number of significant digits to print
    pretty (bool)    If true print the number as 12.345(67).
                     If false, return 12.345 0.0067
                     pretty option not implemented yet

    Returns:
    A string
    """

    vex=orderOfMag(float(val))
    eex=orderOfMag(float(err))

    val /= 10**vex
    err /= 10**eex

    big = max(vex, eex)
    val /= 10**(big-vex)
    err /= 10**(big-eex)

    strr = printe(val, err, sigdig=sigdig, pretty=pretty)
    if pretty:
        strr = "%se%02i" %(strr, big)
    else:
        tmp = strr.split()
        strr= "%se%02i %se%02i" %(tmp[0], big, tmp[1], big)

    return strr


def printe(val, err, sigdig=2, pretty=True):
    """Return a value and its uncertainty in SI format, 12.345(67)

    Input
    val, err (float) The value and it's associated uncertainty
    sigdig (int)     Number of significant digits to print
    pretty (bool)    If true print the number as 12.345(67).
                     If false, return 12.345 0.0067

    Returns:
    A string
    """

    val = float(val)
    err = float(err)

    if err == 0:
        strr = "%.6f 0" %( val)
        return strr

    #Significant digits in the value
    sd = int(max(0, sigdig-1-orderOfMag(err)))


    if pretty:
        if err > 1 and err < 10:
            strr = " %.*f(%.1f)" %(sd, val, err)
        else:
            if sd !=0:
                err /= pow(10, orderOfMag(err)-1)
            #err /= pow(10, orderOfMag(err)-2)
            strr = "%.*f(%g)" %(sd,val,round(err))
    else:
        if err > 1 and err < 10:
            strr = " %.*f %.1f" %(sd, val, err)
        else:
            #err /= pow(10, orderOfMag(err)-1)
            #strr = " %.*f(%g)", sd, val, math.rint(err)
            strr = "%.*f %.*f" %(sd,val,sd,err)

    return strr


def orderOfMag(val):
    """Return the order of magnitude of value

    e.g orderOfMag(9999) is 3

    Input
    val (float)

    Return
    (int)
    """

    return int(math.floor(math.log10(math.fabs(float(val)))))


def respace(stringList, sep=None):
    """Take a list of strings and add white space so the columns
    are left justified.
    """

    #Measure the maximum width of each column
    size = dict()
    for line in stringList:
        if re.search("^#", line):
            continue    #Skip comments

        words = line.split(sep)

        for i, w in enumerate(words):
            if i in size:
                size[i] = max(size[i], len(w))
            else:
                size[i] = len(w)

    #Loop through a second time, format-printing each element
    outList = []
    for line in stringList:
        if re.search("^#", line):
            outList.append(line)  #Don't reformat comments
        else:
            outLine = []
            for i, w in enumerate(line.split(sep)):
                newWord= "%-*s" %(size[i], w)
                outLine.append(newWord)
            outList.append(" ".join(outLine))

    return outList



def decToSeg(ra, dec, sep=" "):
    """Convert ra and dec to a segasdecimal string

    Input
    ra float
    dec float

    Returns
    A string
    """

    #Just in case, force ra and dec to be floats
    ra = float(ra)
    dec= float(dec)

    #First the ra
    des = ra*86400/360. #convert to arcsec
    hr = math.floor(des /3600.)
    mn = math.floor( (des - hr*3600)/60)
    sc = des - hr*3600 -mn*60
    ra = "% 3.0f%s%02.0f%s%05.2f" %(hr, sep, mn, sep,sc)

    #Now the declination
    if dec>0:
        deg = math.floor(dec)
        sign="+"
    else:
       deg = math.ceil(dec)
       sign = "-"

    minsec = abs(dec - deg)
    deg = abs(deg)

    mn = math.floor(minsec*60)
    sc = 3600*(minsec - mn/60)
    dec = "%s%02.0f%s%02.0f%s%04.1f" %(sign, deg,sep, mn,sep, sc)

    return "%s %s" %(ra,dec)


def segToDec(seg,  sep="\s+"):
    """Convert a string of 6 numbers to a decimal ra and dec

    Input
    seg A string of the form hh mm ss.ss +dd mm ss.
    A string with 6 numbers separated by whitespace will do, this function
    does it's best to handle it.

    Returns
    A tuple of (ra, dec)
    """

    a1,a2,a3,a4,a5,a6 = re.split(sep, seg, maxsplit=6)

    #Calculate ra in decimal hours and then convert
    ra = float(a1)
    ra += float(a2)/60.
    ra += float(a3)/3600
    ra *= 360/24.

    a4 = float(a4)
    if a4 < 0:
        dec = -1*a4
        dec += float(a5)/60.
        dec += float(a6)/3600
        dec *= -1
    else:
        dec = a4
        dec += float(a5)/60.
        dec += float(a6)/3600

    return (ra, dec)




def testInRange(val, lwr=None, upr=None, msg=""):
    """Tests that val in in the range [lwr, upr].

    Returns an empty string if true, or a string if false.
    If either lwr or upr are None, this test is not performed.
    """

    fail=0
    if lwr is not None:
        if val < lwr:
            fail = 1

    if upr is not None:
        if val > upr:
            fail = 1

    if fail == 1:
        return "%s: %g not in range [%g, %g] " %(msg, val, lwr, upr)
    return ""



def findOverlap( left, right, output="All"):
    """Find the elements that exist in both sets, or one but not the other

    Inputs
    left
    right   Any iterable object e.g a list or tuple
    output  (string) All|Left|Right|Both
            Left: return a set of all objects in left, but not in right (A')
            Right: return a set of all objects in right, but not in left (B')
            Both: return a set of all objects in both left and right (A ^ B)
            All: Return a tuple of (Left, Both, Right)

    Return
    A set object, or a tuple, depending on the value of output
    """

    A = set(left)
    B = set(right)

    if output == "Left":
        return A - B
    elif output == "Right":
        return B - A
    elif output == "Both":
        return A & B
    else:
        return (A-B, A & B, B-A)


def sortedIndices(self, data):
    """Neat way to do np.argsort() on non numpy lists
    Taken from http://code.activestate.com/recipes/306862/
    """
    return sorted(range(len(data)), key = data.__getitem__)


def createHeader(headerStr, textList=None, columnNames=None, **kwargs):
    """Create a useful header for a text file

    Inputs:
    ------------
    headerStr
        (str) A one line summary of the file contents
    textList
        (list) A list of strings providing additional documentation
    columnNames
        (list) A list of strings giving meta data on the columns

    Options Inputs:
    ----------------
    Any optional arguments are written to the header as key:value
    For example:
    printHeader(... temp="26", units="celcius") produces a header that
    includes the lines
    temp: 26
    units: celcius

    To pass a dictionary of parameters do
    params=dict()
    ...
    printHeader(... **params)

    Returns:
    -------------
    An array of strings.

    Notes:
    -------------
    Function searches for __versio  n__ and __URL global variables in
    your file and prints those to the header if available.
    """
    frame = inspect.currentframe().f_back
    if frame.f_globals['__name__'] == 'kepler.tools':
        frame = frame.f_back

    try:
        idStr = frame.f_globals['__version__']
    except KeyError:
        idStr = frame.f_globals['__file__']

    try:
        urlStr = frame.f_globals['__URL__']
    except KeyError:
        urlStr = "<__URL__ not set>"

    try:
        frameInfo = inspect.getframeinfo(frame)
        funcName = frameInfo[2]
        lineno = frameInfo[1]
    except KeyError:
        funcName = "<Unknown>"
        lineno = "0"

    out = []
    out.append("#%s" %(headerStr))
    out.append("#Created by: %s" %(idStr))
    out.append("#%s" %(urlStr))
    out.append("#Function name: %s" %(funcName))
    out.append("#Line Num: %s" %(lineno))
    out.append("#At: %s" %(datetime.datetime.now()))

    if textList is not None:
        for text in textList:
            out.append("#%s" %(text))

    if columnNames is not None:
        out.append("#Column definitions:")
        for i, line in enumerate(columnNames):
            out.append("#Col %i: %s" %(i, line))

    if len(kwargs.keys()) > 0:
        out.append("#Parameters used")
    for k in kwargs.keys():
        out.append("#%s: %s" %(k, kwargs[k]))
    out.append("#")

    if columnNames is not None:
        words = map( lambda x: x.split()[0], columnNames)
        out.append("#" + " ".join(words))

    return out


def printHeader(fp, headerStr, textList=None, **kwargs):
    """Print a useful header to file pointed to by fp

    Inputs:
    fp          (String or file). If str, function attempts to open
                and write to a file of that name. If file objects,
                then must be writable
    headerStr   (str) A one line summary of the file contents
    textList    (list) A list of strings providing additional documentation

    Options Inputs:
    Any optional arguments are written to the header as key:value
    For example:
    printHeader(... temp="26", units="celcius") produces a header that
    includes the lines
    temp: 26
    units: celcius

    To pass a dictionary of parameters do
    params=dict()
    ...
    printHeader(... **params)

    Returns:
    File pointer. Either fp, or the file pointer to the file called
    fp depending on the type of the input

    Output:
    Text is written to the file pointed to by fp, and fp is kept open.

    Notes:
    Function searches for __versio  n__ and __URL global variables in
    your file and prints those to the header if available.
    """

    if not hasattr(fp, 'write'):
        fp = open(fp, 'w')

    fp.write("\n".join( createHeader(headerStr, textList, **kwargs)))
    fp.write("\n")

    return fp

def compareDicts(dict1, dict2, tupleArray):
    """Compare the values in dict1 to dict2.

    Inputs:
    dict1, dict2:   (Dicts) Dictionaries to be compared
    tupleList       (List) A list of tuples

    Return:
    An empty string if dicts agree. An diagnostic string if they don't.

    Notes:
    Each tuple consists of (key1, key2, function, arg)
    The value of dict1[key1] is compared to dict2[key2].

    If function=None, the keys are compared with a string comparison
    Otherwise, function is called with the arguments
    (dict1[key1], dict2[key2], arg)

    For example, to compare the values as floating point numbers with
    a tolerance of 1e-6 use the tuple (key1, key2, checkClose, 1e-6)

    The comparison function should return True if the values are
    equal in the desired sense, and False if they disagreee
    """

    msg=[]
    for (k1, k2, ptest, arg) in tupleArray:
        if ptest is None:
            if str(dict1[k1]) != str(dict2[k2]):
                msg.append("strComp: %s( %s ) != %s( %s ): " %(k1, dict1[k1], k2, dict2[k2]))
        else:
            err = ptest(dict1[k1], dict2[k2], arg)
            if  err != "":
                testName = (str(ptest).split())[1]
                msg.append("%s: %s( %s ) != %s( %s ) [%s] " %(testName, k1, dict1[k1], k2, dict2[k2], err))

    return msg



def hypotUnc(a, da, b, db):
    """Compute the hypotenuse and its uncertainty"""

    y = np.hypot(a,b)

    c1 = a*da
    c2 = b*db
    dy = np.hypot(c1, c2)/ y

    return y, dy


def haversine(x):
    """Return the haversine of an angle

    haversine(x) = sin(x/2)**2, where x is an angle in radians
    """
    y = .5*x
    y = math.sin(y)
    return y*y


def sphericalAngSep(ra0, dec0, ra1, dec1, radians=False):
    """
        Compute the spherical angular separation between two
        points on the sky.

        //Taken from http://www.movable-type.co.uk/scripts/gis-faq-5.1.html

        NB: For small distances you can probably use
        sqrt( dDec**2 + cos^2(dec)*dRa)
        where dDec = dec1 - dec0 and
               dRa = ra1 - ra0
               and dec1 \approx dec \approx dec0
    """

    if radians==False:
        ra0  = math.radians(ra0)
        dec0 = math.radians(dec0)
        ra1  = math.radians(ra1)
        dec1 = math.radians(dec1)

    deltaRa= ra1-ra0
    deltaDec= dec1-dec0

    val = haversine(deltaDec)
    val += math.cos(dec0) * math.cos(dec1) * haversine(deltaRa)
    val = min(1, math.sqrt(val)) ; #Guard against round off error?
    val = 2*math.asin(val)

    #Convert back to degrees if necessary
    if radians==False:
        val = math.degrees(val)

    return val


def sphericalAngBearing(ra0, dec0, ra1, dec1, radians=False):
    sin = math.sin
    cos = math.cos
    atan  = math.atan2

    if radians==False:
        ra0  = math.radians(ra0)
        dec0 = math.radians(dec0)
        ra1  = math.radians(ra1)
        dec1 = math.radians(dec1)

    dLong = ra1 - ra0
    a = sin(dLong)*cos(dec1)
    b = cos(dec1)*sin(dec1) - sin(dec0)*cos(dec1)*cos(dLong)
    bearing = atan(a, b)

    if radians==False:
        bearing = math.degrees(bearing)

    return bearing


def sphericalAngDestination(ra0_deg, dec0_deg, bearing_deg, dist_deg):
    sin = math.sin
    cos = math.cos
    asin  = math.asin
    atan2 = math.atan2

    phi1 = math.radians(dec0_deg)    #Latitude
    lambda1 = math.radians(ra0_deg)    #Longitude
    d = math.radians(dist_deg)      #Distance in radians
    theta = math.radians(bearing_deg)

    phi2 = sin(phi1)*cos(d)
    phi2 += cos(phi1)*sin(d)*cos(theta)
    phi2 = asin(phi2)

    a = sin(theta)*sin(d)*cos(phi1)
    b = cos(d) - sin(phi1)*sin(phi2)
    #print "A=%.7f b=%.7f" %(a,b)
    #import pdb; pdb.set_trace()
    lambda2 = lambda1 + atan2(a,b)

    ra2_deg = math.degrees(lambda2)
    dec2_deg = math.degrees(phi2)
    return ra2_deg, dec2_deg

