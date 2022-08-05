from ast import For
from ipdb import set_trace as idebug
from typing import Any
import pandas as pd
import numpy as np

import pytest

from stratos.pipeline import  LinearPipeline, BranchingPipeline
from stratos.task import Task




def yes(x):
    return True 

def no(x):
    return False


class BoolTask(Task):
    def __init__(self, value):
        self.value = value 

    def func(self) -> bool:
        return self.value 

class Create(Task):
    def __init__(self, num):
        self.num = num

    # def func(self, arg1:bool) -> list:
    #     return list(range(self.num))
    def func(self, val:bool) -> list:
        return list(range(self.num))

class Sum(Task):
    def func(self, values:list) -> np.int64:
        return np.sum(values)


def test1():

    p1 = LinearPipeline( [Create(4), Sum()] )
    p2 = LinearPipeline( [Create(8), Sum()] )


    pipeline = BranchingPipeline(p1, p2, yes)
    pipeline.validate(True)
    result = pipeline.run(True)
    assert result == 6

    pipeline = BranchingPipeline(p1, p2, no)
    pipeline.validate(True)
    result = pipeline.run(False)
    assert result == 28



def check_is_true(value):
    return value 


def test2():
    p1 = LinearPipeline( [Create(4), Sum()] )
    p2 = LinearPipeline( [Create(8), Sum()] )

    pipeline = [
        BoolTask(True), 
        BranchingPipeline( p1, p2, check_is_true)
    ]
    pipeline = LinearPipeline(pipeline)
    
    pipeline.validate()
    result = pipeline.run()
    assert result == 6

    # pipeline = [
    #     BoolTask(False), 
    #     BranchingPipeline( p1, p2, check_is_true)
    # ]
    # pipeline = LinearPipeline(pipeline)
    # pipeline.validate()
    # result = pipeline.run()
    # assert result == 0
    

def test3():
    p1 = LinearPipeline( [BoolTask(True), Create(4), Sum()] )
    p1.validate()