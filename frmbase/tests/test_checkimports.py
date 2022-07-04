

import frmbase.checkimports as ci
from pprint import pprint 
import os 


def test_do_check_imports():

    objs = ci.do_check_imports("./test_checkimports.py")
    keys = set(map(lambda x: x.split('/')[-1], objs.keys()))
    expected=  set("test_checkimports.py checkimports.py pprint.py os.py".split())
    assert expected <= keys, expected - keys
    pprint(objs)

    objs = ci.do_check_imports("./example_module.py")

    #Check "foo" is in output, that the value is False, then remove from dict
    assert objs.pop('foo') == False 
    keys = set(map(lambda x: x.split('/')[-2], objs.keys()))
    expected=  set("pandas numpy".split())
    assert expected <= keys, expected - keys



def test_find_module_names_in_file():
    mods = ci.find_module_names_in_file('example_module.py')

    print(mods)
    assert set(mods) == set("numpy pandas foo".split()), mods


def test_get_path_to_module():
    path = ci.get_path_to_module('numpy')
    tokens = path.split('/')
    assert tokens[-1] == "__init__.py"
    assert tokens[-2] == "numpy"
    assert tokens[-3] == "site-packages"