from typing import get_type_hints, List
from pprint import pprint
from typeguard import check_type

def check_input_args(func, *args):
    hints = get_type_hints(func)
    pprint(hints)

    input_args = dict(zip(func.__code__.co_varnames, args))
    for key in input_args:
        pprint(key) #, input_args[key], hints[key])
        pprint(input_args[key])
        pprint(hints[key])
        check_type("'"+key+"'", input_args[key], hints[key])




def repeat(a: int, b: List[str]) -> str:
    pass


def main():
    check_input_args(repeat, 1, 'x y z'.split())
