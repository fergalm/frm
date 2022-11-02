from  stratos.pipeline import Pipeline, ForEachPipeline, LinearPipeline
from stratos.task  import Task

from typing import List 
from glob import glob

import pandas as pd

class FindFiles(Task):
    def __init__(self, path):
        self.path = path 

    def func(self) -> List[str]:
        flist = glob(self.path)
        assert len(flist) > 0
        print("Found %i files" %(len(flist)))
        return flist


class LoadFile(Task):
    def func(self, fn: str) -> pd.DataFrame:
        df = pd.read_csv(fn)
        return df

class Head(Task):
    def func(self, df:pd.DataFrame) -> pd.DataFrame:
        return df.head()

class Filter(Task):
    def func(self, df:pd.DataFrame) -> pd.DataFrame:
        #df = df[~df.Winner.isna()].copy()
        df[df.val > 4000].copy()
        return df 

class Concat(Task):
    def func(self, args:List) -> pd.DataFrame:
        return pd.concat(args)


def main():
    #pattern = "/home/fergal/data/elections/MdBoEl/Anne_Arundel/*Primary.csv"
    pattern = "/home/fergal/data/peak_day_forecast/pseg/forecasts/2018/*.csv"

    sub = [ 
        ('load', LoadFile()),
        ('filter', Filter(), 'load'),
        ('head', Head(), 'filter')
    ]
    sub = ForEachPipeline(sub)

    #tasks = [
        #('find', FindFiles(pattern)),
        #('process', sub, 'find'),
        #('cat', Concat(), 'process')
    #]

    tasks = [FindFiles(pattern), sub, Concat()]
    pipeline = LinearPipeline(tasks)
    return pipeline.run()
    
