from ast import Assert
from ipdb import set_trace as idebug
import matplotlib.pyplot as plt 
from pprint import pprint 
import pandas as pd
import numpy as np

import pytest
import dags

from typing import Any

Task = dags.Task
Pipeline = dags.Pipeline
LinearPipeline = dags.LinearPipeline

class A(Task):
    def func(self) -> int:
        return 5 

class B(Task):
    def func(self, val: int) -> np.ndarray:
        return np.arange(val)

class C(Task):
    def func(self, arr:np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame()
        df['val'] = arr
        return df

class Merge(Task):
    def func(self, a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame()
        df['val1'] = a
        df['val2'] = B
        return df


def test_for_islands():
    pipeline = [
        ('a', A()),
        ('b', B()),
        ('c', C(), 'b')
    ]

    with pytest.raises(AssertionError):
        pipeline = Pipeline(pipeline), "Failed to find island"


    pipeline = [
        ('a', A()),
        ('b', B(), 'a'),
        ('c', C(), 'b')
    ]

    pipeline = Pipeline(pipeline)


def test_linear_pipeline():
    
    pipeline = [
        ('a', A()),
        ('b', B(), 'a'),
        ('c', C(), 'b')
    ]

    pipeline = Pipeline(pipeline)
    pipeline.validate()
    df = pipeline.run()
    assert isinstance(df, pd.DataFrame)


def test_diamond_pipeline():
    
    pipeline = [
        ('a', A()),
        ('b1', B(), 'a'),
        ('b2', B(), 'a'),
        ('d', Merge(), 'b1', 'b2')
    ]

    pipeline = Pipeline(pipeline)
    pipeline.validate()
    df = pipeline.run()
    assert isinstance(df, pd.DataFrame)


def test_pipeline_as_task1():
    sub = [
        ('b', B(),),
        ('c', C(), 'b')
    ]
    sub = Pipeline(sub)

    pipeline = [
        ('a', A()),
        ('s', sub, 'a')
    ]

    pipeline = Pipeline(pipeline)
    pipeline.validate()
    df = pipeline.run()


def test_pipeline_as_task2():
    sub = [
        ('a', A()),
        ('b', B(), 'a'),
    ]
    sub = Pipeline(sub)

    pipeline = [
        ('s', sub),
        ('c', C(), 's'),
    ]
    pipeline = Pipeline(pipeline)
    pipeline.validate()
    df = pipeline.run()



def test_linear_pipeline_class():
    pipeline = [A(), B(), C()]
    pipeline = LinearPipeline(pipeline)

    pipeline.validate()
    df = pipeline.run()
    assert isinstance(df, pd.DataFrame)
