from ipdb import set_trace as idebug
from pprint import pprint
from glob import glob 
import pkgutil 
import ast


def get_import_hashes(path, objs=None):
    if objs is None:    
        objs = dict()

    if path in objs or is_builtin_module(path):
        #Already visited, or is builtin module
        return objs 
    else:
        objs[path] = get_hash(path)
        process_module(path, objs)
    return objs 


def process_module(path, objs):
    sub_modules = find_module_names_in_file(path)

    for modname in sub_modules:
        mpath = get_path_to_module(modname)
        if mpath == "":
            continue 

        #Recursively search that module
        subdict = get_import_hashes(mpath, objs)
        objs.update(subdict)
    return objs

def get_path_to_module(modname):
    #Get path of module
    try:
        mpath = pkgutil.get_loader(modname)
        mpath = mpath.get_filename()
    except (AttributeError, ImportError) as e:
        print(f"WARN: Importing {modname} from {path} failed with error {e}")
        return ""


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


def alt_find_module_names_in_file(path):
    """
    
    This doesn't do any better than original yet"""
    modules = []
    text = get_text(path)
    if len(text) == 0:
        return modules

    for line in text:
        words = line.split()
        if words[0] == 'import':
            modname = words[1]
        elif words[0] == 'from' and words[2] == 'import':
            if words[1] == '.':
                #Do something 
                pass 
            elif words[2] == '..':
                #Do something
                pass
            modname = ".".join([words[1], words[3]])
        modules.append(modname)
    return modules




def get_module_name(astNode):
    try:
        modname = astNode.module 
    except AttributeError:
        modname = None 

    #Sometimes happens without throwing exception
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
        #Shared object lib, not python code
        return ""


def is_builtin_module(path):
    #Placeholder code
    if 'site-packages' in path:
        return True

    if 'python3.' in path:
        return True 
    return False




# def find_import_statements(path):
#     with open(path) as fp:
#         text = fp.readlines()

#     modules = []
#     for line in text:
#         words = line.split()
#         if words[0] == 'import':
#             modname = words[1]

#         if words[0] == 'from' and words[2] == 'import':
#             if words[1] == '.':
#                 #Do something 
#                 pass 
#             elif words[2] == '..':
#                 #Do something
#                 pass
#             modname = ".".join([words[1], words[3]])
#         modules.append(modname)
#     return modules

