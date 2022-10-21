
from glob import glob 
import shutil
import time
import os 

"""
A wrapper for running experiments. 

I often want to run a piece of code multiple times with slightly 
different input parameters and compare the results. I'm usually
fiddling with the code at the time, and I want to be able to 

1. Preserve the exact state of the code 
2. Preserve the values of any input parameters
3. Prevent myself from over-writing an old experiment with a new run.

The first goal is met with git or another source code manager. 
The second goal is met with frm.meta, that saves the git revision of the
code, along with any ancillary data. The third goal is met with this code.

Run your code, and once it's done, move all the files it created to
a directory called expt_name, If you try to over-write expt_name, the 
code will prevent you from doing so, unless it has marked that the
previous run threw an exception.

Typical Usage
expt_name = "experiment42"
run_expt(expt_name, my_func, arg1, arg2, expt_name=expt_name)

By passing in the expt_name as a keywork we make it available to my_func,
should my_func want to do something with that value (e.g write it to
a configuration file)


"""

DUMMY_NAME = ".temp_awihge"

def run_expt(expt_name, func, *args, **kwargs):

    if expt_name is None:
        expt_name = DUMMY_NAME

    error_filename = os.path.join(expt_name, "EXPT_DID_NOT_COMPLETE")
    check_expt_exists(expt_name, error_filename)
    
    # before_state = set(glob("*"))
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        #move_new_files(before_state, expt_name)
        move_new_files(expt_name, start_time)
        touch(error_filename)
        raise e 

    #move_new_files(before_state, expt_name)
    move_new_files(expt_name, start_time)
    # _move_new_files_by_modification_time(expt_name, start_time)
    
    if expt_name == DUMMY_NAME:
        remove_old_expt(expt_name)   #Clean up the temporary directory
    return result


def check_expt_exists(expt_name, error_filename):
    if os.path.exists(expt_name):
        if os.path.exists(error_filename):
            #Old expt exists, but should be overwritten
            remove_old_expt(expt_name)
        else:
            #Refuse to overwrite this expt
            raise ValueError("Expt %s already exists" %(expt_name))

def remove_old_expt(expt_name):
    shutil.rmtree(expt_name)


# def move_new_files_by_existence(before, expt_name):
#     after = set(glob("*"))
#     new_files = after - before
#     new_files = filter(lambda x: x[-2:] == "py", new_files)  #Ignore py files
#     os.mkdir(expt_name)

#     for f in new_files:
#         os.rename(f, os.path.join(expt_name, f))


# def _move_new_files_by_modification_time(expt_name, start_time):
def move_new_files(expt_name, start_time):
    """Move files to the expt folder if their modification time is after some point

    This code isn't tested, and probably shouldn't be used yet.
    """
    files = set(glob("*"))
    files = filter(lambda x: x[-2:] != "py", files)  #Ignore py files
    os.mkdir(expt_name)

    for f in files:
        if os.path.getmtime(f) > start_time:
            os.rename(f, os.path.join(expt_name, f))

def touch(path):
    with open(path, 'w') as fp:
        pass 



import pytest
def test_no_over_write():
    os.mkdir("_expt")

    with pytest.raises(ValueError):
        run_expt("_expt", good_func, 1, 2)
    os.rmdir("_expt")

def test_overwrite_failed_expt():
    try:
        run_expt("_expt", failing_func, 1, 2)
    except ZeroDivisionError:
        pass

    #check that a directory was created
    assert os.path.exists("_expt")

    #This should work
    run_expt("_expt", good_func, 1, 2)
    assert os.path.exists(os.path.join("_expt", "tmp.txt"))

    #Cleanup
    shutil.rmtree("_expt")

def good_func(arg1, arg2):
    with open("tmp.txt", "w") as fp:
        fp.write("This is a temporary file")
    return 42

def failing_func(arg1, arg2):
    return 1/0