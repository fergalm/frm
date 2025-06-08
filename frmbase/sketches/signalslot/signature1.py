import inspect
import typing

def check_signatures_match(func, template):
    signature = get_function_signature(func)

    if len(signature) != len(template):
        return False

    for s, t in zip(signature, template):
        if s != t:
            print(f"{signature} does not match {template}")
            return False
    return True


def get_function_signature(func: typing.Callable):
    spec = inspect.getfullargspec(func)
    print(spec)
    #import ipdb; ipdb.set_trace()
    size = len(spec.args)
    annotations = spec.annotations
    f = lambda x: annotations.get(x, typing.Any)
    signature = list(map(f, spec.args))
    return tuple(signature)



def test_get_function_signature1():
    def foo1(x: int, y:float):
        pass

    expected = (int, float)
    assert get_function_signature(foo1) == expected

    def foo2(x: typing.List, y:list):
        pass
    #Note, we have no way of comparing typing.List to list
    expected = (typing.List, list)
    assert get_function_signature(foo2) == expected


    def foo3(x, y:typing.Tuple):
        pass
    expected = (typing.Any, typing.Tuple)
    print(get_function_signature(foo3))
    assert get_function_signature(foo3) == expected

    def foo4(x:list, y:typing.Tuple, *args):
        pass


def test_get_function_signature2():
    def foo1(x=1, y:float=2.5):
        pass
    expected = (typing.Any, float)
    assert get_function_signature(foo1) == expected

    def foo1(x, y:float=2.5):
        pass
    expected = (typing.Any, float)
    assert get_function_signature(foo1) == expected


    def foo2(x: int, y:float, *args):
        pass
    expected = (int, float)
    assert get_function_signature(foo2) == expected

    def foo3(x: int, y:float, **kwargs):
        pass
    expected = (int, float)
    assert get_function_signature(foo2) == expected
