from ipdb import set_trace as idebug
import matplotlib.pyplot as plt 
from pprint import pprint 
import pandas as pd
import numpy as np

import pytest
from  stratos.task import  Task, ValidationError

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

class Sum(Task):
    def func(self, num1:int, num2:int) -> int:
        return num1 + num2

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
    with pytest.raises(ValidationError):
        t1.can_depend_on(t2)


def test_can_depend_on_two():
    """Task C depends on A and B. Test the code does this correctly"""
    t1 = Length()
    t2 = Length()
    t3 = Sum()

    assert t3.can_depend_on(t1, t2)

    t4 = Count()
    with pytest.raises(ValidationError):
        t3.can_depend_on(t1, t4)



def test_validation_with_zero_args():
    t = NullTask()
    t.run()


class Task1a(Task):
    def func(self) -> None:
        return 

class Task1b(Task):
    def func(self):
        return 

class Task2(Task):
    def func(self, a:str):
        return a 

def test_different_num_args():
    t1a = Task1a()
    t1b = Task1b()
    t2 = Task2()

    #t1b doesn't specify a return type, so we default
    #to Any. I may change the behaviour later to treat Any 
    #as a subtype of everything
    with pytest.raises(ValidationError):
        t2.can_depend_on(t1b)

    #t1a explictly tells us it returns 
    with pytest.raises(ValidationError):
        t2.can_depend_on(t1a)
    
