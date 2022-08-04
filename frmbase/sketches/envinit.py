from pdb import set_trace as debug
import sys
import os 

def main():
    """
    Sketch of an idea of a python script that would read a config
    file and set env appropriately.

    This would be called from a bash script.

    setup.src::

        import ~/exl/default.src
        cmd conda deactivate
        cmd conda activate my_env
        append PYTHONPATH ./my_py

    ::
        > eval envinit

    """
    if len(sys.argv) == 2:
        script = sys.argv[1]
    else:
        script = find_config_script()
    cmds = build(script)
    
    print( "\n".join(cmds))


def find_config_script(path=None):
    ps = '/' 
    
    if path is None:
        path = os.getcwd()
    else:
        path = os.path.realpath(path)
        
    elts = path.split(ps)
    for i in range(len(elts), 1, -1):
        trial_path = ps.join(elts[:i])
        trial_path = os.path.join(trial_path, "setup.src")
        
        if os.path.exists(trial_path):
            return trial_path 
    
    raise IOError(f"No file setup.src found in directory tree above {path}")


def build(fn):

    funcs = {
        'import': import_func,
        'append': append_func,
        'conda': conda_func,
        'unset': unset_func,
        'set': set_func,
        'cmd': cmd_func,
        '#': ignore_comment,
    }

    with open(fn) as fp:
        text = fp.readlines()

    output = []
    for line in text:
        words = line.split()
        
        if words[0][0] == '#':
            words[0] = '#'
        
        try:
            func = funcs[words[0]]
        except IndexError:
            #Blank line
            continue 
        except KeyError:
            raise KeyError(f"Command {words[0]} not understood")

        output.extend(func(words))
    return output

def unset_func(words):
    write_me()

def set_func(words):
    #I can't remember what I wanted this func to do
    write_me()
    
def cmd_func(words):
    return [" ".join(words[1:])]
#Write a decorator to ensure outputs are always lists
#@checkoutputs
def import_func(words):
    srcfile = words[1]

    if os.path.exists(srcfile):
        return build(srcfile)
    else:
        raise IOError(f"Src file {srcfile} not found")

def conda_func(words):
    return [" ".join(words)]

def append_func(words):
    var = words[1]
    vals = words[2:]
    
    out = []
    for v in vals:
        out.append(f"export {var}=${var}:{v}")
    return out 

#@checkoutputs
def ignore_comment(words):
    return []

def write_me(words):
    raise ValueError(f"{words[0]} is a legal command that is not yet implemented")


if __name__ == "__main__":
    main()
