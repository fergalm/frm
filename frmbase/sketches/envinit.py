
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

    """"
    cmds = build(sys.argv[1])
    print( "\n".join(cmds))

def build(fn):

    funcs = {
        'import': import_func,
        'append': write_me,
        'conda': write_me,
        'unset': write_me,
        'set': write_me,
        'cmd': write_me,
        '#': ignore_comment,
    }

    with open(fn) as fp:
        text = fp.readlines()

    output = []
    for line in text:
        words = line.split()
        try:
            func = funcs[words[0]]
        except KeyError:
            raise KeyError(f"Command {words[0]} not understood")

        output.extend(func(words))
    return output


#Write a decorator to ensure outputs are always lists
#@checkoutputs
def import_func(words):
    srcfile = words[1]

    if os.path.exists(srcfile)
        return build(srcfile)
    else:
        raise IOError(f"Src file {srcfile} not found")
    
#@checkoutputs
def ignore_comment(words):
    return []

def write_me(words):
    raise ValueError(f"{words[0]} is a legal command that is not yet implemented")