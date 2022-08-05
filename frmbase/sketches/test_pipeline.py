from re import L
from xml.dom import ValidationErr
from ipdb import set_trace as idebug
from typing import Any
import pandas as pd
import numpy as np

import pytest

from stratos.pipeline import Pipeline, LinearPipeline
from stratos.task import Task, ValidationError
import stratos.task  as task

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


def test_for_duplicate_labels():
    pipeline = [
        ('a', A()),
        ('b', B(), 'a'),
        ('a', C(), 'b')  #Note duplicate label 
    ]

    with pytest.raises(KeyError):
        pipeline = Pipeline(pipeline)


def test_for_validation_fail():
    pipeline = [   #Dependencies in the wrong order.
        ('a', A(), 'b'),
        ('b', B(), 'c'),
        ('c', C(), )
    ]

    pipeline = Pipeline(pipeline)
    with pytest.raises(ValidationError):
        pipeline.validate()



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
    """This is failing because task d validates against b1 and b2 seperately"""
    
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


# This is future work
# def test_initial_inputs():
#     pipeline = [('b', B(), )]
#     pipeline = Pipeline(pipeline)
    
#     pipeline.validate(5)
#     result = pipeline.run(5)
#     assert np.allclose(result, np.arange(5))

def test_pipeline_as_task1():
    """This isn't perfect. The sub pipeline must be indepdently validated"""
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
    # df = pipeline.run()


def test_pipeline_as_task0():
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


def test_validate_pipeline_with_input_arg():
    pipeline = Pipeline( [ ('b', B(),)] )
        
    assert pipeline.validate(4)
    with pytest.raises(ValidationError):
        pipeline.validate()

    with pytest.raises(ValidationError):
        pipeline.validate('a')
    
    with pytest.raises(ValidationError):
        pipeline.validate(4, 5)
        

def test_linear_pipeline_class():
    pipeline = [A(), B(), C()]
    pipeline = LinearPipeline(pipeline)

    pipeline.validate()
    df = pipeline.run()
    assert isinstance(df, pd.DataFrame)
