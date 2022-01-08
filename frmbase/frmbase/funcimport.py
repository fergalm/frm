
import importlib

def import_function(func_name):
    """Parse input func_name into a function object

    Example
    -----------
    `import_function('os.path.exists')` is equivalent to 

    ```
    import os.path
    f = os.path.exists
    return f
    ```

    This can also be achieved with an eval, but an eval is more dangerous
    because it can execute arbitrary code.
    """
    tokens = func_name.split('.')
    module_name = ".".join(tokens[:-1])
    f_name = tokens[-1]

    try:
        _module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError("Module %s not found" %(module_name))

    try:
        _func = getattr(_module, f_name)
    except AttributeError:
        raise ValueError("Function %s not found in module %s" %(f_name, module_name))

    return _func