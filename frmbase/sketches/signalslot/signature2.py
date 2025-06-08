from typing import Callable, Tuple, Any
import inspect

"""

class Foo:
    signal1 = Signal()

    def __init__(self):
        self.signal2 = Signal( (int, str) )

    def foo():
        ...
        self.signal1.emit(StrangeObject())
        self.signal2.emit(1, "one")


def slot1(obj):
    print("Running slot1")

def slot2(integer, string):
    print("Running slot2")


def main():
    f1 = Foo()
    f2 = Foo()

    Foo.signal1.connect(slot1)
    f1.signal2.connect(slot2)
    #f1.signal2.connect(slot1)  #Would raise ValueError


    #Prints "running slot1" and "running slot2"
    f1.foo()

    #Prints "running slot1" only
    f2.foo()


Tests
Create and accept
    nothin g
    int
    int str
    ArbitraryObject
Reject the wrong kind of slot func

Test class level and object level variables

"""




class Signature:
    """
    TODO
    --------
    Implement __lt__, to say that once signature is
    compatible with another
    """

    def __init__(self, signature, allow_args, allow_kwargs):
        self.signature = signature
        self.allow_args = allow_args
        self.allow_kwargs = allow_kwargs

    @classmethod
    def from_function(cls, func:Callable):
        spec = inspect.getfullargspec(func)

        size = len(spec.args)
        annotations = spec.annotations
        f = lambda x: annotations.get(x, Any)
        signature = list(map(f, spec.args))

        allow_args = spec.varargs is not None
        allow_kwargs = spec.varkw is not None
        return cls(signature, allow_args, allow_kwargs)

    def __eq__(self, other):
        if not isinstance(other, Signature):
            return False

        if self.allow_kwargs != other.allow_kwargs:
            return False

        if self.allow_args != other.allow_args:
            return False

        if len(self.signature) != len(other.signature):
            return False

        for s, t in zip(self.signature, other.signature):
            if s != t:
                print(f"{self} does not match\n{other}")
                return False
        return True

    def __repr__(self):
        msg = [
            f"<Signature: {self.signature},",
            f"allow_args={self.allow_args},"
            f"allow_kwargs={self.allow_kwargs}"
            ">"
        ]
        return " ".join(msg)

class Signal:
    """
    A signal is like an exception which doesn't halt
    code execution.

    It is useful to allow a piece of code encounters
    an event it doesn't know how to deal with. The
    signal tells the calling code that something has happened
    but then carries on.

    A fuller description is in the module level documentation
    """

    def __init__(self, signature:Signature | Tuple = None):

        if isinstance(signature, Tuple):
            signature = Signature(signature, False, False)
        self.signature = signature

        self.listeners = []

    def __repr__(self):
        if self.signature is None:
            sig = "not defined"
        else:
            sig = self.signature

        return f"<Signal with signature: {sig}>"

    def connect(self, func: Callable):
        """Add a function to listens to this signal.

        When the signal emits, every listening function
        will get called in sequence
        """

        func_sig = Signature.from_function(func)

        if self.signature != None:
            if func_sig != self.signature:
                raise ValueError(f"Slot {func} does not match sig {self.signature}")
        self.listeners.append(func)

    def emit(self, *args, **kwargs):
        """Call all the listeners and tell them an event happened

        Inputs
        ----------
        All arguments are passed to the listeners

        Returns
        -----------
        **None**
        """
        for func in self.listeners:
            func(*args, **kwargs)



import pytest
def test_sig0():

    def foo0():
        pass

    def foo1(x):
        pass

    sig = Signal(tuple())
    sig.connect(foo0)
    with pytest.raises(ValueError):
        sig.connect(foo1)


def test_sigany():

    def foo0():
        pass

    def foo1(x):
        pass

    sig = Signal((Any,))
    sig.connect(foo1)
    with pytest.raises(ValueError):
        sig.connect(foo0)




def test_sigint_str():

    def foo1(x:int):
        pass

    def foo2(x:str):
        pass

    sig = Signal((int,))
    sig.connect(foo1)
    with pytest.raises(ValueError):
        sig.connect(foo2)


def test_sigarb():
    class Foo:
        pass

    def func1(x:Foo):
        pass

    def func2(x:int):
        pass

    sig = Signal((Foo,))
    sig.connect(func1)
    with pytest.raises(ValueError):
        sig.connect(func2)


def test_class_level():

    class Worker:
        class_signal = Signal()

        def __init__(self, name):
            self.name = name
            self.object_signal = Signal()

        def apply(self):
            self.class_signal.emit(self.name)
            self.object_signal.emit(self.name)

    def slot1(caller):
        print(f"Slot 1 called by {caller}")

    def slot2(caller):
        print(f"Slot 2 called by {caller}")

    w1 = Worker("W1")
    w2 = Worker("W2")

    #Connect all workers to slot 1
    Worker.class_signal.connect(slot1)

    #Connect a single worker to slot 2
    w1.object_signal.connect(slot2)

    w1.apply()
    w2.apply()



#Tests
#Create and accept
    #nothin g
    #int
    #int str
    #ArbitraryObject
#Reject the wrong kind of slot func

#Test class level and object level variables
