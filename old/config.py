# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pdb import set_trace as debug
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib as mpl
import pandas as pd
import numpy as np

from collections import OrderedDict
import importlib
import os

"""
A cascading configuration object.

This configuration object aims to solve the problem of loading
configuration of multiple locations. For example, packageA may depend
on packageB and packageC. packageA needs a way to load configuration data
from packageC, then from packageB, over-writing values if necessary. packageA
also needs a way to record the correct order of packages to load.

Simple Uses
-------------
cfg = Config('ExampleConfigurationObject')
cfg['param1'] = 'Spam'
cfg['param1']
>>> 'Spam'

cfg.set('param1', 'Ham', 'Spamalot')
cfg.get('param1'')
>>> 'Ham'

#Get this history of who set param1. '__setitem__' means parameter
#was set using square bracket notation.
cfg.getProvenance('param1')
>>> ['Spamalot', '__setitem__']

#Add a dictionary of values
bundle = {'param2':'Herring', 'param3':'Shrubbery'}
cfg.setFromDict(bundle, 'Nih')
cfg.getProvenance('param2')
>>> ['Nih']


Loading from Json
----------------
You can specify a set of cascading json files to load configuration data
from.

...


"""

class Config(object):
    """A cascading configuration object

    A container two dictionaries. One stores a set of configuration values,
    the other stores a list of locations that previously set this value.
    This provenance makes debugging a configuration setup easier because
    you can see both where a value was set, and who else is trying to set
    that same value.

    The use case is that you configuration pulls setup information from
    multiple packages. Package A may set 'defaultServer' to 'exampleA.com',
    while package B sets the same value to 'exampleB.com'. The config class
    loads values from package A, then overwrite them with values with those
    from package B. It then maintains a list of places that set or updated
    the value of 'defaultServer'

    Most of the implementation can be understood by looking at the ``set()``
    method.


    """
    def __init__(self, name):
        """
        TODO
        -----
        Optionally take a config file as argument
        """

        self.values = OrderedDict()
        self.source = OrderedDict()
        self.name = name


    def __getitem__(self, key):
        return self.get(key)


    def __setitem__(self, key, value):
        """Set a value using a dictionary call, eg confg['key'] = value

        There is no way to specify the provenance of a key with this approach,
        so it should be used carefully. The provenence is set to '__setitem__'
        """
        self.set(key, value, '__setitem__')


    def set(self, key, value, source, overwrite=True):
        """Set the value of configuration parameters.

        Inputs
        ---------
        key
            (str) Name of parameter
        value
            (string or number) Value of parameter
        source
            (string) Provenance. Metadata to indicate who is setting this value

        Optional Inputs
        ---------------
        overwrite
            (bool) If **False**, method refuses to over write a value.

        Returns
        ----------
        **None**

        Note, no type checking is done on the inputs. Caveat Emptor.
        """

        if not overwrite and key in self.values:
            raise ValueError("%s already present. Set overwrite=True to replace")

        self.values[key] = value

        if key in self.source:
            self.source[key].append(source)
        else:
            self.source[key] = [source]


    def setFromEnv(self, key, envVar=None, overwrite=True):
        """Set a parameter based on a shell envirnomental variable

        Inputs
        ---------
        key
            (str) Name of key to set in config object. By default
            this is also the name of the variable in the shell
            environment.

        Optional Inputs
        -----------------
        envVar
            (str) If not **None**, use this environment variable instead
        overwrite
            (bool) If **False**, refuse to over write a value.

        Returns
        ----------
        **None**
        """
        if envVar is None:
            envVar = key

        value = os.environ[envVar]
        self.set(key, value, "Environment", overwrite)


    def setFromDict(self, dictionary, name, overwrite=True):
        """Read in parameters from a dictionary

        Inputs
        ----------
        dictionary
            (dict) Dictionary of parameters
        name
            (str) Provenance of values in this dictionary.

        Optional Inputs
        -----------------
        overwrite
            (bool) If **False**, refuse to over write a value.

        Returns
        ----------
        **None**
        """

        for key in dictionary:
            self.set(key, dictionary[key], name, overwrite)


    def merge(self, config, overwrite=True):
        """Merge contents of `config` into this object, where
        `config` is an object of type Config"""
        dictionary = config.values
        name = config.name

        self.setFromDict(dictionary, name, overwrite)


    def get(self, key, defaultValue=None):
        """Retrieve value of parameter

        Optional Inputs
        ---------------
        defaultValue
            If not **None** return this value if `key` not set, otherwise
            raises an exception.
        """
        if key not in self.values:
            if defaultValue is None:
                raise KeyError("Key %s not set in configuration object %s" \
                               %(key, self.name))
            else:
                return defaultValue

        return self.values[key]


    def getProvanence(self, key):
        """Gets the provancence of a parameter.

        The provenance is the history of who set the value of `key`.
        Knowing this can be helpful for debugging.

        Returns
        --------
        A list
        """

        if key not in self.values:
            raise KeyError("Key %s not set in configuration object %s" \
                           %(key, self.name))

        return self.source[key]


    def loadYaml(self, fn):
        with open(fn) as fp:
            obj = oyaml.load(fp)

        if 'name' not in obj:
            raise IOError("Input file %s not a valid json for Config (name is missing)")

        #Recursively load configuration from modules
        if 'modules' in obj:
            for mod in obj['modules']:
                try:
                    moduleObject = importlib.import_module(mod)
                except ImportError:
                    raise ValueError("Can't find %s on python path" %(mod))
                path = moduleObject.__path__[0]
                configObject = Config(mod)
                configObject.loadYaml(path + "/config.yaml")
                self.merge(configObject)

        #Load parameters
        if 'parameters' in obj:
            name = obj['name']
            parameters = obj['parameters']
            self.setFromDict(parameters, name)

    def display(self):
        """Pretty Print the parameters"""
        names = self.values.keys()
        values = self.values.values()
        sources = self.source.values()

        nameLen = max(map( len, names))
        valLen = max(map(lambda x: len(str(x)), values))
        for n, v, s in zip(names, values, sources):
            print("%-*s: %-*s (%s)" % \
                  (nameLen, n, valLen, v, ",".join(s)) )



#    import oyaml
#    def toYaml(self, fn):
#        """
#        Writes a new config file. Forgets all proveneance.
#
#        TODO
#        -----
#        Write some meta data to the comments
#        """
#        text = []
#        text.append("name: %s\n" %(self.name))
#        text.append("parameters:")
#
#        for k in self.values:
#            text.append("    %s: %s" %(k, str(self.values[k])))
#
#        with open(fn, 'w') as fp:
#            fp.write("\n".join(text))
