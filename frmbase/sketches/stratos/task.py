# from types import NoneType
from ipdb import set_trace as idebug
from pprint import pprint 
import pandas as pd

from typing import Any, List, get_type_hints
from typeguard import check_type
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
    """Base of a task class.

    Any function you want to run in a pipeline should be wrapped in 
    a Task object. For example::

        def foo(data:):
            return data 

        class FooTask(Task):
            #Overide this method with your function logic
            #Type hints should always be used for this method definition
            def func(self, data:np.ndarray) -> np.ndarray:  
                return foo(data)

    The Task class provides two useful features. 
    1. It provides methods to access the type hint signature of `func()`
       so that the pipeline can check if consecutive classes are 
       compatible in terms of the types of the objects passed between them.
       This enables a pipeline to check that its components are compatible
       before the pipeline is run.

    2. It checks that the actual values input and returns are consistent
       with the type hints at run time. This ensures the logic honours the
       promises made in the type hints.

    """
    def __call__(self, *args):
        return self.run(*args)

    def get_input_signature(self) -> List:
        #This fails if an argument isn't decorated
        type_hints = get_type_hints(self.func)
        type_hints.pop('return', None)  #Remove return value if present
        return list(type_hints.values())

    def get_output_signature(self) -> List:
        type_hints = get_type_hints(self.func)
        return_type = [type_hints.pop('return', Any)]
        return return_type

    def run(self, *args):
        print("Running %s with args %s" %(self.name(), args)) 
        validate_args(args, self.get_input_signature())
        result = self.func(*args)
        validate_args(result, self.get_output_signature())
        return result 

    def validate(self):
        return True

    def can_depend_on(self, *args):
        """Can this task accept output of dependent tasks

        Input
        --------
        A list of task objects.

        Returns
        ---------
        **Bool**

        Check that the list return types of the provided tasks matches the input signature of our called
        function .

        """
        sig1 = self.get_input_signature()

        sig2 = []
        for task in args:
            sig = task.get_output_signature()
            if not isinstance(sig, list):
                sig = [sig]
            sig2.extend(sig)

        msg = f"Task\n  {self}\n  expects\n  {sig1}\n  but input tasks\n  {args}\n  supplies\n  {sig2}"
        if len(sig1) != len(sig2):
            raise ValidationError(msg)

        for a, b in zip(sig1, sig2):
            # idebug()
            try:
                check_type('', a, b)
            except TypeError:
                raise ValidationError(msg)
        return True 
        
    def func(self, df: pd.DataFrame) -> str:
        """Overwrite this function with task logic"""
        return self.name()

    def name(self):
        name = str(self).split()[0][1:]
        return name 


def validate_args(actual, expected):
    if not isinstance(actual, tuple):
        actual = [actual]
    if len(actual) != len(expected):
        raise ValidationError(f"Expected {len(expected)} arguments, got {len(actual)}")

    for act, exp in zip(actual, expected):
        try:
            check_type("", act, exp)  #Throws an exception
        except TypeError:
            msg = f"Expected {exp}, but type of {act} is {type(act)}"
            raise ValidationError(msg)



class GenericTask(Task):
    """Wrap a pre-defined function in a task 
    
    Most tasks are just simple wrappers around a predefined function.
    Creating a task in this generic, most common, case is handled
    by this class

    """

    def __init__(self, func, *args, **kwargs):
        """*args and **kwargs are the configuration arguments to the function.

        Example
        ---------::

            def power(x, n, int_only=False):
                ...

            task = GenericTask(power, 2, int_only=False)
            task.run(4) #--> 16
        """
        self.func = func 
        self.args = args 
        self.kwargs = kwargs 

    def run(self, *params):
        print("Running %s with args %s" %(self.name(), params)) 
        validate_args(params, self.get_input_signature())
        result = self.func(*params, *self.args, **self.kwargs)
        validate_args(result, self.get_output_signature())
        return result 


# def create_task(func, *args, **kwargs):
#     """Create a task from a function
#
#     The fixed arguments in args and kwargs are passed to the function
#     call after any arguments generated by the pipeline. For example::
#
#         task2 = create_task(func, 1 ,2, b=4)
#         pipeline = LinearPipeline([task1, task2])
#
#     `func` will be called as::
#
#         func(x1, 1, 2, b=4)
#
#     where `x1` is the output of `task1`. On the other hand, for a pipeline
#     like ::
#
#         pipeline = Pipeline(
#             [ ('t0', task0),
#               ('t1', task1,
#               ('t2', task2, ['t0', 't1']),
#             ]
#         )
#
#     Then func will be called as::
#
#         func(x0, x1, 1, 2, b=4)
#     """
#     return GenericTask(func, *args, **kwargs)