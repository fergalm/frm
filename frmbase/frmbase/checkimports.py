from ipdb import set_trace as idebug
from pprint import pprint
from glob import glob
import pkgutil
import ast

"""
Code to recusively seach a python module for import 
statements. This can be useful when you want
to package up a new piece of code and want to 
find all the dependencies

TODO
Fails on imports in the style ::

    from . import foo
    from .. import foo
"""


def check_imports_for_dir(path):
    """Check all imports for all python files in path

    TODO:
    Recursively search subdirectories of path

    """
    flist = glob(path + "/*.py")
    assert len(flist) > 0

    objs = dict()
    for f in flist:
        objs.update(do_check_imports(f, objs))
    print_report(objs)


def check_imports(path):
    """Given the path to a file recursively check for modules the need to be imported.
    This is useful when packaging new code.
    """
    objs = do_check_imports(path)
    print_report(objs)


def print_report(objs):
    missing = lfilter(lambda x: objs[x] is False, objs.keys())
    reqs = sorted(list(objs.keys()))

    print("The following imports were found")
    pprint(reqs)

    print("ERROR: Failed to import these modules")
    pprint(missing)


def do_check_imports(path, objs=None):
    if objs is None:
        objs = dict()

    if path in objs:
        # Already visited
        return objs
    else:
        objs[path] = True

    if not is_builtin_module(path):
        process_module(path, objs)
    return objs


def process_module(path, objs):
    sub_modules = find_module_names_in_file(path)

    for modname in sub_modules:
        mpath = get_path_to_module(modname)
        if mpath == "":
            print(f"WARN: Importing {modname} from {path} failed.")
            objs[modname] = False
            continue 

        #Recursively search that module
        subdict = do_check_imports(mpath, objs)
        objs.update(subdict)
    return objs

def get_path_to_module(modname):
    #Get path of module
    try:
        mpath = pkgutil.get_loader(modname)
        mpath = mpath.get_filename()
    except (AttributeError, ImportError) as e:
        return ""
    return mpath    

def find_module_names_in_file(path):
    """
    
    This fails for imports of type::

        from . import foo
        from .. import foo
    """
    modules = []
    text = get_text(path)
    if len(text) == 0:
        return modules

    nodes = ast.parse(text)
    for node in ast.walk(nodes):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            modules.append(get_module_name(node))
    return modules



def get_module_name(astNode):
    try:
        modname = astNode.module
    except AttributeError:
        modname = None

    # Sometimes happens without throwing exception
    if modname is None:
        modname = astNode.names[0].name
    return modname


def get_hash(path):
    text = get_text(path)
    return hash(text)


def get_text(path):
    try:
        with open(path) as fp:
            return fp.read()
    except (TypeError, UnicodeDecodeError):
        # Shared object lib, not python code
        return ""


def is_builtin_module(path):
    # Placeholder code
    if "site-packages" in path:
        return True

    if "python3." in path:
        return True
    return False


def lfilter(func, vals):
    return list(filter(func, vals))
