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

    def run(self, *args):
        print("Running %s with args %s" %(self.name(), args)) 
        self.validate_input_args(args)
        result = self.func(*args)
        self.validate_return_value(result)
        return result 

    def validate_input_args(self, args:List):
        hints = self.get_input_signature()

        argcount = self.func.__code__.co_argcount
        argnames = self.func.__code__.co_varnames[:argcount]
        if argnames[0] == 'self':
            argnames = argnames[1:]

        pprint(locals())
        for name, val in zip(argnames, args):
            validate_type(val, hints[name])

    def get_input_signature(self):
        hints = dict()
        annotations = self.func.__annotations__  #Mnuemonic
        argcount = self.func.__code__.co_argcount
        argnames = self.func.__code__.co_varnames[:argcount]

        for arg in argnames:
            hints[arg] = annotations.get(arg, Any)
        # pprint(locals())
        # idebug()
        hints.pop('self', None)
        return hints 

    def validate_return_value(self, ret_val):
        hint = self.get_output_signature()
        validate_type(ret_val, hint)

    def get_output_signature(self):
        # idebug()
        hint = self.func.__annotations__.get('return', Any)
        if hint is None:
            hint = type(None)
        return hint 

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
        my_hints = self.get_input_signature()
        my_hints.pop('self', None)

        # idebug()
        if len(args) != len(my_hints):
            msg = f"Task {self} expected {len(my_hints)} args, got {len(args)}"
            raise ValidationError(msg)

        for key, task in zip(my_hints.keys(), args):
            if not is_compatible(task.get_output_signature(), my_hints[key]):
                raise ValidationError
        return True

    def validate(self):
        return True

    def func(self, df: pd.DataFrame) -> str:
        """Overwrite this function with task logic"""
        return self.name()

    def name(self):
        name = str(self).split()[0][1:]
        return name 

def validate_type(val, hint):
    try:
        check_type('', val, hint)
    except TypeError:
        msg = f"Type of {val} is {type(val)}, but I expected {hint}"
        raise ValidationError(msg)


def is_compatible(class_type, hint):
    if hint is Any:
        return True 
    
    #TODO: This means that a function that does not declare a return
    #type can't be accepted as input to a function that declares a type
    #for its inputs. Is this what I want?
    if class_type is Any:
        return False 
    return issubclass(class_type, hint)

# def validate_args(actual, expected):
#     if not isinstance(actual, tuple):
#         actual = [actual]
#     if len(actual) != len(expected):
#         raise ValidationError(f"Expected {len(expected)} arguments, got {len(actual)}")

#     for act, exp in zip(actual, expected):
#         try:
#             check_type("", act, exp)  #Throws an exception
#         except TypeError:
#             msg = f"Expected {exp}, but type of {act} is {type(act)}"
#             raise ValidationError(msg)



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
        raise NotImplementedError()
        #I need to think about validation here
        print("Running %s with args %s" %(self.name(), args)) 
        self.validate_input_args(params)
        result = self.func(*params, *self.args, **self.kwargs)
        self.validate_return_value(result)
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