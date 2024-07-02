
from pathlib import Path
import numpy as np 
import requests
import os 
import re 

"""
Download precinct level results for Maryland elections
from Maryland Board of Elections website
(elections.maryland.gov/elections)

To add results for a new year, change the
value of `end` in `main()` and re-run. Previously
downloaded files will be skipped.


Note that 2020 primary precinct results not available
on BoEl website
"""

def main():

    dl =MdBoElResultDownloader()
    # dl.downloadPrimary(2022, "All", "Republican")
    end = 2024
    for year in np.arange(2014, end+1, 2):
        dl.downloadGeneral(year, "Baltimore")
        dl.downloadPrimary(year, "Baltimore", "Democratic")
        dl.downloadPrimary(year, "Baltimore", "Republican")

        dl.downloadGeneral(year, "All")
        dl.downloadPrimary(year, "All", "Democratic")
        dl.downloadPrimary(year, "All", "Republican")

    #     # dl.downloadPrimary(year, "All", "Democratic")
    #     # dl.downloadPrimary(y, "Republican", "All")
    #     # dl.downloadPrimary(y, "Republican", "Baltimore")

    #     # dl.downloadGeneral(year, "All")
    
    
class MdBoElResultDownloader:
    def __init__(self):
        self.url = "https://elections.maryland.gov/elections"
        self.useCache = True 
        self.basePath = "."

        self.countyCodes = {
            'All': 'All',
            'Baltimore': '04'
        }

    def downloadGeneral(self, year, region):
        fn = self.getGeneralFilename(year, region)
        path = self.url, f"{year}", "election_data", fn
        path = "/".join(path)

        outfile = self.getSavePath(year, "General", region)
        status = self.downloadFile(path, outfile)
        if status != 200:
            path = re.sub("PrecinctResults", "Precincts", path)
            status = self.downloadFile(path, outfile)
        

    def downloadPrimary(self, year, region, party):
        fn = self.getPrimaryFilename(year, party, region)
        path = self.url, f"{year}", "election_data", fn
        path = "/".join(path)

        outfile = self.getSavePath(year, "Primary", region, party)
        status = self.downloadFile(path, outfile)
        if status != 200:
            path = re.sub("PrecinctResults", "Precincts", path)
            status = self.downloadFile(path, outfile)
        

    def getPrimaryFilename(self, year, party, region):
        if year < 2021:
            return f"{region}_By_Precinct_{party}_{year}_Primary.csv"
        else:
            return self.getNewStyleFilename(year, "P", region, party)

    def getGeneralFilename(self, year, region):
        if year < 2021:
            return f"{region}_By_Precinct_{year}_General.csv"
        else:
            return self.getNewStyleFilename(year, "G", region)

    def getNewStyleFilename(self, year, estage, region, party=None):
        party = party or ""

        estage = estage[0].upper()
        if estage not in ['P', 'G']:
            raise ValueError("Estage must be either 'P' for primary or 'G' for General")
            
        etype = self.getEtype(year)
        code = self.countyCodes[region]
        return f"{etype}{estage}{year-2000}_{code}PrecinctResults{party}.csv"
        
        # PP24_AllPrecinctsDemocratic.csv
        # GG22_AllPrecincts.csv

    def getEtype(self, year):
        if (year-2000) % 4 == 0:
            etype = "P"  #Presidential year
        elif (year-2000) % 4 == 2:
            etype = "G"  #Gubernatorial year
        else:
            raise ValueError(f"No election for {year}")
        return etype

    def getSavePath(self, year, eStage, region, party=None):
        eStage = eStage[0].upper()
        if eStage == 'P':
            assert party is not None
            name = f"{region}_PrecinctResults_{year}_Primary_{party}.csv"    
        elif eStage == 'G':
            name = f"{region}_PrecinctResults_{year}_General.csv"    
        else:
            raise ValueError(f"Election stage {eStage} not known")

        path = os.path.join(self.basePath, region, name)
        return path

    def downloadFile(self, url, localPath):
        print(url)
        print(localPath)
        print("+++")
        # return 
        if self.useCache and os.path.exists(localPath):
            print(f"INFO: Skipping {url}")
            return 200
        
        basepath = Path(localPath).parent
        os.makedirs(basepath, exist_ok=True)

        res = requests.get(url)
        if res.status_code != 200:
            print(f"WARN: failed to download {url}")
            return res.status_code
        
        with open(localPath, 'w') as fp:
            fp.write(res.text)

        return res.status_code


