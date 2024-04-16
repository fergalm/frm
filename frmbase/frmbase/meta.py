
"""
Routines to save meta data relating to the outputs of a code.
Write a json file containing all the local variables in a function as well
as information about the run date, and git status of the file.

This informtion can be useful to a Data Scientist running many similar experiments.
You can record the exact file version and set of parameters used to create a data
set alongside the results files produced in an experiment

"""


from ipdb import set_trace as idebug
from typing import Dict
import datetime
import inspect
import json

import pandas as pd
import os
import re

try:
    import git 
    _GIT_PYTHON_INSTALLED = True
except ImportError:
    _GIT_PYTHON_INSTALLED = False
    pass 


class MyJsonEncoder(json.encoder.JSONEncoder):
    """If an object to be serialised has a __json__ method, use 
    that for serialisation. Otherwise try default encoding,
    and fall back on a string representation
    """

    def default(self, obj):
        if hasattr(obj, '__json__'):
            if callable(obj.__json__):
                return obj.__json__()
            else:
                return obj.__json__ 
        try:
            return super(MyJsonEncoder, self).default(obj)
        except TypeError:
            return str(obj)


def save_state(statefile, **kwargs):
    """Write out the values local variables to a json file

    A quick and dirty way of writing out the parameters used
    in a run to a file to document what exactly you were doing.

    Inputs
    -------------
    statefile
        (string) Path of file to write to. Using a .json extension
        is recommented (e.g file.json)

    Optional Inputs
    ---------------
    Any optional inputs are written to the output json file
    as key,value pairs.

    Returns
    ------------
    **None**

    Output
    -----------
    The local variables of the calling function, the optional
    arguments, and a few other useful bits of information are
    written to ``statefile`` in json format

    Example
    -----------
    ::

        numSim = 1000
        alpha = .5
        meta = save_state('run1.json', comment='Initial guess at alpha')
        run_big_sim(numSim, alpha)

    Notes
    -----------
    * Intented use is to write out strings, ints, floats and other
    small objects. json may reject things like classes and arrays,
    or may produce large, unreadable output. If you have such data
    you may perfer to use `save_metadata` instead.

    * Classes can be saved if you add a __json__() method to convert
    the class to a dict
    """
    frame = kwargs.pop('frame', inspect.currentframe().f_back)
    # params = dict(frame.f_locals)

    # kwargs.update(params)
    jsonstr = get_state(frame=frame, **kwargs)

    with open(statefile, 'w') as fp:
        fp.write(jsonstr)


def save_metadata(statefile, **kwargs):
    """Save some metadata to a json file

    `save_state` saves the local state of a function. Sometimes
    you don't want the local variables (because they don't persist
    easily), but just want to save the metadata. That is what
    this function is for.

    Inputs
    -------------
    statefile
        (string) Path of file to write to. Using a .json extension
        is recommented (e.g file.json)

    Optional Inputs
    ---------------
    Any optional inputs are written to the output json file
    as key,value pairs.


    Returns
    ------------
    **None**

    Output
    -----------
    The optonal arguments, as well as some information about the
    calling function and file, are
    written to ``statefile`` in json format

    Example
    -----------
    .. code-block:: python

        numSim = 1000
        alpha = .5
        save_metadata('run1.json', numSim=numSim, alpha=.5,
                   comment='Initial guess at alpha')
        run_big_sim(numSim, alpha)

    Notes
    -----------
    * Intented use is to write out strings, ints, floats and other
    small objects. json may reject things like classes and arrays,
    or may produce large, unreadable output.

    * See also `save_state`
    """
    frame = kwargs.pop('frame', inspect.currentframe().f_back)
    jsonstr = get_metadata(frame=frame, **kwargs)

    with open(statefile, 'w') as fp:
        fp.write(jsonstr)


def get_state(**kwargs) -> str:
    """Get the state of local variables.

    This function performs the same function as save_state, but
    returns a dictionry of parameters instead of saving the file to disk

    Returns
    ------------
    A dictionary

    Example
    -----------
    .. code-block:: python

        numSim = 1000
        alpha = .5
        meta = get_state('run1.json', comment='Initial guess at alpha')
        run_big_sim(numSim, alpha)

    Notes
    -----------
    * Intented use is to write out strings, ints, floats and other
    small objects. json may reject things like classes and arrays,
    or may produce large, unreadable output. If you have such data
    you may perfer to use `save_metadata` instead.
    """

    frame = kwargs.pop('frame', inspect.currentframe().f_back)
    params = dict(frame.f_locals)

    kwargs.update(params)
    return get_metadata(frame=frame, **kwargs)


def get_metadata(**kwargs) -> str:
    """Save some metadata to a dictionary

    This function performs the same function as get_state, but
    returns a dictionry of parameters instead of saving the file to disk


    Returns
    ------------
    A dictionary

    Example
    -----------
    :: 

        numSim = 1000
        alpha = .5
        meta = get_metadata('run1.json', numSim=numSim, alpha=.5,
                   comment='Initial guess at alpha')
        run_big_sim(numSim, alpha)

    Notes
    -----------
    * Intented use is to write out strings, ints, floats and other
    small objects. json may reject things like classes and arrays,
    or may produce large, unreadable output.

    * See also `save_state`
    """

    #frame is the function that called save_metadata.
    #`save_state` passes in the frame that called it as an optional argument
    #so we use that if available.
    frame = kwargs.pop('frame', inspect.currentframe().f_back)
    params = dict()

    (filename, lineno, funcname, _, _) = inspect.getframeinfo(frame)
    params['__file__'] = filename
    params['__func__'] = funcname
    params['__lineno__'] = lineno
    params['__date__'] = str(datetime.datetime.now())

    try:
        params['__user__'] = os.environ['USER']
    except KeyError:
        params['__user__'] = None

    if _GIT_PYTHON_INSTALLED:
        try:
            params.update(get_git_info(filename))
        except (IOError, ValueError):
            params['__git_remote_url__'] = "NotAGitRepository"

    #Add docstring from calling function
    name = frame.f_code.co_name
    docstring = frame.f_globals[name].__doc__    
    if isinstance(docstring, str):  #If no docstring, will be None
        docstring = docstring.split('\n')
    params['__doc__'] = docstring

    params.update(kwargs)
    return json.dumps(params, indent=2, cls=MyJsonEncoder)



def get_git_info(filename) -> dict:
    params = dict()

    if filename[0] != '/' and filename[1] != ':':
        raise ValueError("get_git_info requires a full path")

    try:
        repo = get_git_repo(filename)
    except IOError:
        params['git_remote_url'] = "NotAGitRepository"
        return params

    branch = repo.active_branch
    commit = branch.commit

    params['__git_remote_url__'] = repo.remote().url
    params['__git_branch__'] = branch.name
    date = pd.to_datetime(commit.authored_date, unit='s')
    params['__git_branch_commit_date__'] = str(date)
    params['__git_branch_commit_author__'] = str(commit.author)
    params['__git_branch_commit_sha__'] = commit.hexsha[:8]

    params['__git_commit_status__'] = get_commit_status(repo, filename)
    return params



def get_git_repo(filename) -> str:
    sep = os.path.sep
    tokens = filename.split(sep)

    for i in range(len(tokens)-1, 1, -1):
        try:
            path = sep.join(tokens[:i])
            return git.Repo(path, expand_vars=False)
        except (git.NoSuchPathError, git.InvalidGitRepositoryError):
            continue


    raise IOError("%s does not appear to belong to a git repo" %(filename))




def get_commit_status(repo, filename) -> str:
    #Windows work around. gitpython stores paths with forward slash
    sep = os.path.sep 
    filename = filename.replace(sep, '/')

    #Convert filename from a full path. to one relative
    #to the root directory of the repo, which is what GitPython needs
    repo_dir = os.path.split(repo.git_dir)[0]
    local_path = filename[len(repo_dir)+1:]

    if local_path in repo.untracked_files:
        return "Unstaged"
    else:
        diffs = repo.index.diff(None)
        for d in diffs:
            if local_path in d.a_path:
                return "Modified"
        return "Commited"

