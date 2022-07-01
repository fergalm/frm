from types import NoneType
from ipdb import set_trace as idebug
from pprint import pprint 
import pandas as pd

from inspect import signature
from typing import Any
import inspect 

"""
Every pipeline should be a task, in that it implements Pipeline.run()

Every task should validate its inputs and outputs at run time
Every task should implmeent validation at compile time.
Every pipeline should validate too. 

For validation to be useful, I must be able to validate columns
(and optionally dtypes!) of dataframes, and for the presence
of keys in dicts
"""

class ValidationError(Exception):
    pass 

class Task():

    def __call__(self, *args):
        return self.run(*args)

    def get_input_signature(self):
        annotation_list = [x.annotation for x in signature(self.func).parameters.values()]
        return annotation_list        

    def get_output_signature(self):
        try:
            return self.func.__annotations__['return']
        except KeyError:
            return Any

    def run(self, *args):
        print("Running %s with args %s" %(self.name(), args)) 
        self.validate_args(args, self.get_input_signature())
        result = self.func(*args)
        self.validate_args(result, self.get_output_signature())
        return result 

    def validate_args(self, actual, expected):
        if not isinstance(actual, tuple):
            actual = [actual]

        if not isinstance(expected, list):
            expected = [expected]

        if len(actual) != len(expected):
            raise ValidationError(f"Expected {len(expected)} arguments, got {len(actual)}")

        i = 0
        for act, exp in zip(actual, expected):
            if exp in [Any, inspect._empty]:  #no type hint
                continue 

            if exp is None:
                exp = NoneType

            if not isinstance(act, exp):
                raise ValidationError(f"Argument {i}: Expected {exp}, found {act}")
            i += 1

    def can_depend_on(self, task2):
        sig1 = self.get_input_signature()
        sig2 = task2.get_output_signature()
        if not isinstance(sig2, list):
            sig2 = [sig2]

        for a, b in zip(sig1, sig2):
            if a != b:
                msg = f"Task {self} expects {sig1} but task {task2} supplies {sig2}"
                print(msg)
                return False 
        return True 
        
    def func(self, df: pd.DataFrame) -> str:
        """Overwrite this function with task logic"""
        return self.name()

    def name(self):
        name = str(self).split()[0][1:]
        return name 


