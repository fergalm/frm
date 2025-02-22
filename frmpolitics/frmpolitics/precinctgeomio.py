import numpy as np 
import os 
import frmgis.get_geom as fgg 
import frmbase.dfpipeline as dfp

def loadBalcoPrecinctGeoms(year, basepath=None):
    """Read in the appropriate precinct boundary geometries for a given year.
    
    For example, if year is 2024, the most recent precincts are from 
    2020, so load those 
    
    Inputs
    ---------
    year
        (int) Year of interest
    basepath
        (str) Path where geometry fiels can't be found. If not passed,
        sensible defaults for my laptop will be used
        
    Returns 
    ------------
    A dataframe with columns of Precinct and geom 

    Note
    ---------
    Precincts are formatted as %02i-%03i. I think this is the preferred format
    """
    
    #Paths to use for each precinct file 
    paths = {
        2010: '/home/fergal/data/elections/shapefiles/precinct2014/BaCoPrecinct-wgs84.shp', #2014 data!
        2014: '/home/fergal/data/elections/shapefiles/precinct2014/BaCoPrecinct-wgs84.shp',
        2022: '/home/fergal/data/elections/shapefiles/precinct2022/BNDY_Precincts2022_MDP_WGS84.shp',
    }
    
    #Pipeline to pass results of reading the file, per year
    pipelines = {
        2010: parse2014(),
        2014: parse2014(),
        2022: parse2022(),
    }
    
    availableYears = np.array(list(paths.keys()))
    availableYears = availableYears[availableYears <= year]
    if len(availableYears) == 0:
        raise ValueError(f"Requested year ({year}) is before earliest year {np.min(availableYears)}")
    bestYear = np.max(availableYears)
    
    path = paths[bestYear]
    
    if basepath is not None:
        fn= os.path.split()[-1]
        path = os.path.join(basepath, fn)
    
    df = fgg.load_geoms_as_df(path)
    print(df.iloc[0])
    tasks = pipelines[bestYear]
    df = dfp.runPipeline(tasks, df)
    return df 




def parse2014():
    cols = "NAME geom".split()
    pipeline = [
        dfp.SelectCol(cols),
        dfp.RenameCol({'NAME': 'Precinct'}),
        dfp.DropDuplicates("Precinct"),
    ]
    return pipeline


def parse2022():
    cols = "LABEL geom".split()
    pipeline = [
        dfp.SelectCol(cols),
        dfp.RenameCol({'LABEL': 'Precinct'}),
        dfp.DropDuplicates("Precinct"),
    ]
    return pipeline

