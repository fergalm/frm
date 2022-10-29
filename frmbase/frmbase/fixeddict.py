from ipdb import set_trace as idebug
from ipdb import set_trace as idebug

class FixedDict(dict):
    """A dictionary that doesn't allow new keys to be  accidentlly added.

    Acts as a better base class for special purpose dictionary classes 

    Keys are defined on creation. For example::

        fixed = FixedDict('a', 'b')  #Default values are **None**
        fixed = FixedDict(a=1, b=2)

        d = dict(a=1, b=2, c=3)
        fixed = FixedDict(d)


    The dictionary can then be treated as a normal dictionary, with
    one exception. Attempts to add a new key will fail::

        assert fixed['a'] == 1  
        fixed['a'] = 2    #Works

        try:
            fixed['aa'] = 4  #Fails
        except KeyError:
            raise KeyError("Can't add a new key")

    If you do need to add a new key, use the set method::

        fixed.set('aa', 4)

    One nice feature of FixedDicts is that you can address keys
    like attributes::

        assert fixed.a == 2
        fixed.a == 3

        fixed.aa   #Raises attribute error
    
    TODO
    ------
    * Don't allow the names of dict methods (keys, copy, items, etc) as
      keys to a FixedDict

    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        for a in args:
            if hasattr(a, 'keys'):  #Input is a dictionary
                self.update(a)
            else:
                dict.__setitem__(self, a, None)

        for k in kwargs:
            dict.__setitem__(self, k, kwargs[k])

    # #Not working yet
    # def __getstate__(self):
    #     # idebug()
    #     state = self.__dict__.copy()
    #     return state
    #
    # def __setstate__(self, state):
    #     """Allows the object to be properly unpickled"""
    #     # idebug()
    #     print("The state of you: %s" %(state))
    #     self = state

    def __setitem__(self, key, value):

        if key not in self.keys():
            raise KeyError(f"Can't add new keys to this object ({key})")
        dict.__setitem__(self, key, value)

    def set(self, key, value):
        """Force adding a new key to the dictionary"""
        dict.__setitem__(self, key, value)

    #Gettattr is working
    def __getattr__(self, key):
        """Treat fixeddict.xxx as the same as fixedict['xxx'].

        If that fails, try to look up the attribute normally
        """
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return dict.__getattribute__(self, key)

    # Not working, should forbid setting not dictionary attributes
    def __setattr__(self, key, value):
        try:
            self[key] = value
        except KeyError:
            if hasattr(self, key):
                dict.__setattr__(self, key, value)
            else:
                raise AttributeError
        return


