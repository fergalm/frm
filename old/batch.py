import datetime
import os



def batchrun(func, inputlist, output_generator,
 *args, **kwargs):
    """Run *func* on every file in inputlist, unless it has already been run

    For each file in *inputlist*, use *output_generator*  to create an output
    file. If that file does not already exist, run *func* on the input file.

    This is useful if you are running a large job on many inputs. If it crashes
    for some reason, and you don't want to re-run all the jobs (e.g it crashed
    because of a power failure), run the job through this function.

    Inputs
    ------
    func
        (Callable) The function to run. Signature is `func(file_in, file_out,
        ...)`
    inputlist
        (iterable) List of files to process
    output_generator
        (callable) Computes the output file name based on the input filename.
        See *default_output_generator* for the format


    Optional Inputs
    -----------------
    batch_verbose
        Make the batch function verbose, printing the value of
        inputlist as a progress meter

    All other optional inputs are passed directly to func

    Returns
    --------------
    **None**


    """

    force = kwargs.pop('batch_force', False)
    verbose = kwargs.pop('batch_verbose', False)

    outputlist = map( output_generator, inputlist)

    for f1, f2 in zip(inputlist, outputlist):
        if verbose:
            t0 = datetime.datetime.now()
            print "Processing %s --> %s" % (f1, f2),

        if not os.path.exists(f2) or force:
            func(f1, f2, *args, **kwargs)

        if verbose:
            t1 = datetime.datetime.now()
            print "%s" %(t1-t0)

def default_output_generator(filename):
    """An example output filename generator function

    Input
    ----------
    filename
        (str) Name of input filename


    Returns
    --------------
    Name of output filename
    """
    return os.path.basename(filename) + ".out"




def example_func(filein, fileout, param1, param2, keyword1=True):
    """An example function.

    For this example, the input file is ignored.
    """

    print "Operating on %s" % (filein)

    fp = open(fileout, 'w')
    fp.write("1: %s  2: %s  3: %s" % (param1, param2, keyword1) )
    fp.close()




def test():

    file_list = "a b".split()
    file_list = map( lambda x: x+".txt", file_list)
    out_list = map( default_output_generator, file_list)

    #Create file a.txt
    f1out = out_list[0]
    fp = open(f1out, 'w')
    fp.write("This is some text")
    fp.close()

    mtime1 = os.path.getmtime(out_list[0])

    batchrun(example_func, file_list, default_output_generator,
             'param1', 'param2', keyword1=False)

    #All output files exist
    for f in out_list:
        assert os.path.exists(f)

    #zeroth file wasn't modified
    mtime2 = os.path.getmtime(out_list[0])
    assert mtime1 == mtime2

    for f in out_list:
        os.remove(f)
