from ipdb import set_trace as idebug
from pprint import pprint
import importlib
import ast
import pkgutil 

def check_imports(path, objs=None):
    if objs is None:    
        objs = set()

    if path in objs:
        #Already visited
        return objs 
    process_module(path, objs)

def process_module(path, objs):
    objs |= {path}

    text = get_text(path)
    if len(text) == 0:
        return objs

    nodes = ast.parse(text)
    for node in ast.walk(nodes):
        if isinstance(node, ast.Import) or \
            isinstance(node, ast.ImportFrom):

            modname = get_module_name(node)            
            mpath = pkgutil.get_loader(modname)
            if mpath is None:
                print(f"WARN: In {path} Can't import {modname}")
                idebug()
                continue

            try:
                mpath = mpath.get_filename()
            except AttributeError:
                #Some builtins throw this error
                continue 

            subdict = check_imports(mpath, objs)
            objs.update(subdict)
    return objs



def makeTree(path, objs=None):
    if objs is None:    
        objs = dict()
    
    text = get_text(path)
    if len(text) == 0:
        return objs

    # print(text[:100])
    objs[path] = hash(text)

    nodes = ast.parse(text)
    for node in ast.walk(nodes):
        if isinstance(node, ast.Import) or \
            isinstance(node, ast.ImportFrom):

            modname = get_module_name(node)            
            print(modname)
            mpath = pkgutil.get_loader(modname)
            if mpath is None:
                print(f"WARN: In {path} Can't import {modname}")
                idebug()
                continue

            try:
                mpath = mpath.get_filename()
            except AttributeError:
                #Some builtins throw this error
                continue 

            if mpath in objs or is_builtin_module(mpath):
                continue 

            objs[mpath] = get_hash(mpath)
            subdict = makeTree(mpath, objs)
            objs.update(subdict)
    return objs

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
