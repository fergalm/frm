import numpy as np 
import os 
import frmgis.get_geom as fgg 
import frmbase.dfpipeline as dfp

def loadBalcoPrecinctGeoms(year, basepath=None):
    
    paths = {
        2014: '/home/fergal/data/elections/shapefiles/precinct2014/BaCoPrecinct-wgs84.shp',
        2022: '/home/fergal/data/elections/shapefiles/precinct2022/BNDY_Precincts2022_MDP_WGS84.shp',
    }
    
    pipelines = {
        2014: parse2014(),
        2022: parse2022(),
    }
    
    availableYears = np.array(list(paths.keys()))
    availableYears = availableYears[availableYears <= year]
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
        dfp.ApplyFunc('Precinct', fixPrecinct),
    ]
    return pipeline


def parse2022():
    cols = "LABEL geom".split()
    pipeline = [
        dfp.SelectCol(cols),
        dfp.RenameCol({'LABEL': 'Precinct'}),
        dfp.ApplyFunc('Precinct', fixPrecinct),
    ]
    return pipeline


def fixPrecinct(row):
    return "0" + row.Precinct
