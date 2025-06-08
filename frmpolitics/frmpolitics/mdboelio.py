
"""
MarylanD Board Of ELections IO routines

Routines to read in Board of Elections
results files for every election and 
parse them to have similar columns.

These files are downloaded from BoEl website
using `mdboeldn`. The format of the file
changes slightly from year to year,
so this module has parsers to ensure
that important columns show up with
the same name and format across the years.

I've only done some of the columns. If you
need another column done, make sure you 
make it consistent across all years.

Notes
----------
* Precinct boundaries change between
2020 and 2022.

* 2020-2024 loaders verify that a minimum set of
columns are present. You should add similar logic
to the other loaders

"""

import pandas as pd 
import numpy as np 
import os 

ROOT_PATH = "/home/fergal/data/elections/MdBoEl"

def loadPrecinctResultsForGeneral(year, region='Baltimore'):
    fn = f"{region}_PrecinctResults_{year}_General.csv"
    fn = os.path.join(ROOT_PATH, region, fn)
    df = pd.read_csv(fn)
    df = parseMdBoElPrecinctResults(df, year)
    return df


def loadPrecinctResultsForPrimary(year, party, region='Baltimore'):

    if party[0].lower() == 'D':
        party = "Democratic"
    elif party[0].lower() == 'R':
        party = "Republican"
    else:
        raise ValueError(f"Party {party} not recognised")
    
    fn = f"{region}_PrecinctResults_{year}_Primary_{party}.csv"
    fn = os.path.join(ROOT_PATH, region, fn)
    df = pd.read_csv(fn)
    df = parseMdBoElPrecinctResults(df, year)
    return df


def parseMdBoElPrecinctResults(df, year:int):

    parserDict = {
        2024: parser2022,  #Same as 2024
        2022: parser2022,
        2020: parse2020,
        2018: parse2018,
        2016: parse2014, #No change from 2014
        2014: parse2014,  
    }

    try:
        parseFunc = parserDict[year]
    except KeyError:
        print(f"WARN: Can't find parser for year {year}. Trying default")
    
    return parseFunc(df)




def parser2022(df):
    """Also does double duty as the 2024 parser"""
    mapper = {
        'County': 'CountyFips', 
        'County Name': 'County', 
        'Election District - Precinct': 'Precinct',
        'Election Precinct': 'Precinct',# 'Congressional',
        # 'Legislative',
        'Office Name': 'Office',
        # 'Office District',
        'Candidate Name': 'Candidate',
        # 'Party',
        # 'Winner',
        'Early Votes': 'Early',
        'Early Voting Votes': 'Early',
        'Election Night Votes': 'Doip',
        'Election Day Votes': 'Doip',
        'Mail-In Ballot 1 Votes': 'MailIn1',
        'By Mail Votes': 'MailIn1',
        'Provisional Votes': 'Provisional',
        'Prov. Votes': 'Provisional',
        'Mail-In Ballot 2 Votes': 'MailIn2', 
    }

    df = df.rename(mapper, axis=1)
    df['TotalVote'] = df.Early + df.Doip + df.Provisional + df.MailIn1 
    if 'MailIn2' in df.columns:
        df['TotalVote'] += df.MailIn2
        
    df['Precinct'] = df.Precinct.str[1:]
    validate_df_out(df)
    return df


def parse2020(df):
    mapper = {
        'County': 'CountyFips', 
        'County Name': 'County', 
        'Election District': 'District',
        'Election Precinct': 'Precinct',# 'Congressional',
        # 'Legislative',
        'Office Name': 'Office',
        'Candidate Name': 'Candidate',
        # 'Party',
        # 'Winner',
        'Early Votes': 'Early',
        'Early Voting Votes': 'Early',
        'Election Night Votes': 'Doip',
        'Election Day Votes': 'Doip',
        'Mail-In Ballot 1 Votes': 'MailIn1',
        'By Mail Votes': 'MailIn1',
        'Provisional Votes': 'Provisional',
        'Prov. Votes': 'Provisional',
        'Mail-In Ballot 2 Votes ': 'MailIn2', 
        'By Mail 2 Votes': 'MailIn2',
    }

    def _foo(row):
        return "%02i-%03i" %(row.District, row.Precinct)

    df = df.rename(mapper, axis=1)
    df['Precinct'] = df.apply(_foo, axis=1)
    df['County'] = "Empty"
    df['TotalVote'] = df.Early + df.Doip + df.Provisional + df.MailIn1 + df.MailIn2
    
    validate_df_out(df)
    return df


def parse2018(df):
    mapper = {
        'County': 'CountyFips', 
        'County Name': 'County', 
        'Election District': 'District',
        'Election Precinct': 'Precinct',# 'Congressional',
        # 'Legislative',
        'Office Name': 'Office',
        # 'Office District',
        'Candidate Name': 'Candidate',
        # 'Party',
        # 'Winner',
        'Early Votes': 'Early',
        'Early Voting Votes': 'Early',
        'Election Night Votes': 'DOIP',
        'Election Day Votes': 'DOIP',
        'Mail-In Ballot 1 Votes': 'MailIn1',
        'By Mail Votes': 'MailIn1',
        'Provisional Votes': 'Provisional',
        'Prov. Votes': 'Provisional',
        'Mail-In Ballot 2 Votes ': 'MailIn2', 
    }

    def _foo(row):
        return "%02i-%03i" %(row.District, row.Precinct)

    df = df.rename(mapper, axis=1)
    df['Precinct'] = df.apply(_foo, axis=1)
    df['TotalVote'] = df.DOIP
    df['Office'] = df.Office.str.rstrip()
    return df


def parse2014(df):
    mapper = {
        'County': 'CountyFips', 
        'Election District': 'District',
        'Election Precinct': 'Precinct',# 'Congressional',
        'Office Name': 'Office',
        'Candidate Name': 'Candidate',
        'Election Night Votes': 'DOIP',
    }

    def _foo(row):
        return "%02i-%03i" %(row.District, row.Precinct)

    df = df.rename(mapper, axis=1)
    df['Precinct'] = df.apply(_foo, axis=1)
    df['TotalVote'] = df.DOIP
    return df


def validate_df_out(df):
    cols = df.columns
    
    missing_cols = set(load_schema().keys()) - set(cols)
    if len(missing_cols) > 0:
        msg = f"Expected columns not found: {missing_cols}"
        raise ValueError(msg)

    assert np.all(df.Precinct.str.len() == 6)

def load_schema():
   
   keys = """
        CountyFips
        County
        Precinct 
        Office
        Early 
        Doip
        MailIn1
        MailIn2
        Provisional
        """.split()
   
   out = dict()
   for k in keys:
       out[k] = None 
   return out 


