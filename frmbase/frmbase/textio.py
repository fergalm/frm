"""
Created on Wed Nov 30 11:09:10 2016

Tools to write csv files with useful headers

@author: fergal
"""

import datetime
import inspect
import re
import os

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
    frame = inspect.currentframe()
    try:
        idStr = frame.f_back.f_globals['__version__']
    except KeyError:
        idStr = frame.f_back.f_globals['__file__']

    #funcName = frame.f_back.f_globals['__name__']

    out = []
    out.append("#%s" %(headerStr))
    out.append("#Created by: %s" %(idStr))
    out.append("#User: %s" %(os.environ['USER']))
    #fp.write("#Function name: %s\n" %(funcName))
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
    out.append("")
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

    if not isinstance(fp, file):
        fp = open(fp, 'w')

    fp.write("\n".join( createHeader(headerStr, textList, **kwargs)))
    fp.write("\n")

    return fp
