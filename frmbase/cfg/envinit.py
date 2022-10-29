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
    
    with open(script) as fp:
        print(fp.read())


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


if __name__ == "__main__":
    main()
