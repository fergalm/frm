from ipdb import set_trace as idebug
import matplotlib.pyplot as plt 
from pprint import pprint 
import pandas as pd
import numpy as np

import pytest
from  task import  Task, ValidationError

from typing import Any


class Length(Task):
    def func(self, msg:str) -> int:
        return len(msg)

class Length2(Task):
    def func(self, msg: Any) -> Any:
        return len(msg)

class Length3(Task):
    def func(self, msg) :
        return len(msg)

class Count(Task):
    def func(self, value:int) -> tuple:
        return tuple(range(1, int+1))


class NullTask(Task):
    def func(self) -> None:
        """Does nothing"""
        return 


def test_validation():
    task = Length()

    val = task.run("message")
    assert val == 7

    with pytest.raises(ValidationError):
        val = task.run(7)


def test_validation_any():
    task = Length2()

    val = task.run("message")
    assert val == 7

    #Fails instead of spotting the wrong kind of input
    #because type hint is Any
    with pytest.raises(TypeError):
        val = task.run(7)


def test_validation_no_hints():
    task = Length3()

    val = task.run("message")
    assert val == 7

    #Fails instead of spotting the wrong kind of input
    #because type hint is Any
    with pytest.raises(TypeError):
        val = task.run(7)


def test_can_depend_on():
    t1 = Length()
    t2 = Count()

    assert t2.can_depend_on(t1)
    assert not t1.can_depend_on(t2)



def test_validation_with_zero_args():
    t = NullTask()
    t.run()

