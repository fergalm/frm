from ast import For
from ipdb import set_trace as idebug
from typing import Any
import pandas as pd
import numpy as np

import pytest

from pipeline import  LinearPipeline, ForEachPipeline
from task import Task


class Create(Task):
    def __init__(self, num):
        self.num = num 

    def func(self) -> list:
        # assert arg1 or not arg1
        return list(range(self.num))

class SquareTask(Task):
    def func(self, num:int) -> int:
        return num*num 

class AddOne(Task):
    def func(self, num:int) -> int:
        return num + 1

class Sum(Task):
    def func(self, values:list) -> np.int64:
        return np.sum(values)

def test1():

    # idebug()
    tasks = [ SquareTask()]
    p1 = LinearPipeline(tasks)
    p2 = ForEachPipeline(p1)

    p2.validate()
    result = p2.run([1,2,3,4,5])
    print(result)


def test2():

    #Create a list of numbers, square each one and add one, then
    #sum each element of the list
    t1 = Create(5)
    t2 = LinearPipeline([SquareTask(), AddOne()])
    t2 = ForEachPipeline(t2)

    t3 = Sum()
    p2 = LinearPipeline([t1, t2, t3])

    p2.validate()
    result = p2.run()
    assert result == 35
