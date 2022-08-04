from ipdb import set_trace as idebug
from frmbase.support import lmap
import matplotlib.pyplot as plt 
from pprint import pprint 
from typing import List
import networkx as nx

from .task import Task, ValidationError

import frmbase.flogger 

log = frmbase.flogger.log
log.setLevel(frmbase.flogger.DEBUG)

"""
Todo
o BranchingPipeline is failing its tests 
o Pipeline should be able to turn off garbage collection 
o foreach pipeline should type check its sub-pipeline 
o Doc strings
o A pipeline should provide methods to ensure it has a single starting/ending node
o Check that objects, not classes are passed to Pipeline

"""
class Pipeline(Task):
    """
        A class to run a series of Task objects whose results are used
        to feed other tasks.

        Each task takes some input, and returns some output. The 
        inputs may come from the results of one or more other tasks,
        and the outputs may feed one or more additional tasks. The
        pipeline takes care of ensuring that

        1. Each task that says it depends on a second task expects as 
           input the results of the other second task. 
        2. The tasks are run in the correct order so no task is run
           before its dependent tasks. 

        Pipelines can be linear, e.g ::

            A ---> B ---> C ----> D

        or support branching and merging, e.g 
        ::

            .          C ---> D
            .          |      |
            .A --> B --|      |--> G --> H
            .          |      |
            .          E ---> F


        In this example,
        * A feeds B
        * B feeds both C and E
        * C feeds D, E feeds F
        * Both D and F feed G
        * G feeds H

        Usage
        ------
        Create a pipeline with a list of tuples. Each tuple contains
        * A label (e.g a string)
        * A Task to run for that node
        * An optional list of dependencies

        For example to create the linear pipeline above you can do::

            tasklist = [
                ('a', TaskA()),  #No dependencies listed
                ('b', TaskB(), 'a'),  #Node 'b' depends on results of node 'a'
                ('c', TaskC(), 'b'),
                ('d', TaskD(), 'c'),
            ]

            pipeline = Pipeline(tasklist)

        (Note: The LinearPipeline class in this module makes creating
        linear pipelines like this a little easier).

        For the branched pipeline you would do::
        
            tasklist = [
                ('a', TaskA()) , 
                ('b', TaskB(), 'a'),  
                ('c', TaskC(), 'b'),
                ('d', TaskD(), 'c'),  
                ('e', TaskE(), 'b'), #Tasks c and e depnd on b
                ('f', TaskF(), 'e'),  
                ('g', TaskB(), 'd', 'f'),  #Depends on two tasks 
                ('h', TaskC(), 'g'),

            pipeline = Pipeline(tasklist)

        The pipeline will figure out the order in which the tasks
        must be run and feed the correct outputs to the correct inputs.

        If the tasks have proper typehints on their func() methods,
        the pipeline will check that the inputs and outputs of tasks
        are compatible both before and during the pipeline run. Catching
        incompatible types before running the pipeline can dramatically
        reduce runtime.


        Note on Chaining Pipelines
        ---------------------------

        The Pipeline class extends the Task class, with 
        the goal of treating a pipeline as just another task. The goal
        is that you can do something like::

            p1 = Pipeline(tasklist1)
            p2 = Pipeline(tasklist2)
            p3 = LinearPipeline([p1, p2])

        There are a couple of caveats to bear in mind.

        1. A pipeline that depends on a previous task (or pipeline) should
           have one and only one internal task upon which every other
           task is dependent on. So a pipeline that looks like::

            O-------O
              |
              |-----O

        can depend on a task, but a pipeline like this can not

            O----------O
                  /
            O----/

        For this second graph, the task that is supposed to receive the
        input is ambiguous, and the results are undefined. This may
        fail in unexpected ways.

        2. A pipeline that supplies the input to another task (or pipeline)
           must have one and only one "end" task, that no other task
           depends on. 
    """

    def __init__(self, tasks):
        self.graph, self.task_dict = self.create_graph(tasks)
        self.verify_graph()
        self.tasklist = list(nx.topological_sort(self.graph))[::-1]

    def create_graph(self, tasklist):
        labels = dict()
        g = nx.DiGraph()

        for row in tasklist:
            label, func, deps = row[0], row[1], row[2:]

            if label in labels:
                raise KeyError(f"Two tasks with same label ({label}) found!")
            
            labels[label] = dict(func=func, reqs=row[2:])
            g.add_node(label)
            for d in deps:
                g.add_edge(label, d)
        return g, labels

    def verify_graph(self):
        assert nx.is_directed_acyclic_graph(self.graph)
        assert nx.is_weakly_connected(self.graph), "Islands found in the graph"


    def run(self, *inputs):
        self.validate()

        for i, label in enumerate(self.tasklist):
            func = self.task_dict[label]['func']
            reqs = self.task_dict[label]['reqs']

            f = lambda x: self.task_dict[x]['result']
            
            if i != 0:
                inputs = lmap(f, reqs)

            result = func(*inputs)
            self.task_dict[label]['result'] = result 
            self.garbage_collect(label)
            self.report_status()
        return result

    def validate(self):
        tasklist = self.tasklist[::-1]

        val = True 
        for label in tasklist:
            t0 = self.task_dict[label]['func']
            reqs = self.graph.successors(label)
            log.debug(f"Task {label}({t0})...")
            log.debug(str(t0.get_input_signature()))
            #TODO. I have to collect all req tasks, and pass a list of them
            #into can_depend_on. can_depend_on has to accept a list
            for r in reqs:
                t1 = self.task_dict[r]['func']
                log.debug(f"Comparing to {r}")
                val &= t0.can_depend_on(t1)
            
            if val:
                log.info(f"Task {label} validates")

        if not val:
            raise ValidationError("Validation failed")

    def get_input_signature(self):
        """See note on chaining pipelines"""
        label = self.tasklist[0] 
        func = self.task_dict[label]['func']
        return func.get_input_signature()

    def get_output_signature(self):
        """See note on chaining pipelines"""
        label = self.tasklist[-1] 
        func = self.task_dict[label]['func']
        return func.get_output_signature()

       
    def garbage_collect(self, label):
        """Clear out intermediate results stored in memory
        """
        parents  = list(self.graph.successors(label))

        for par in parents:
            if self.all_children_complete(par):
                #No more need to the cached result, so free that memory
                self.task_dict[par]['result']  = None 
                #Recursively garbage collect the recently freed node
                self.garbage_collect(par)

    def all_children_complete(self, label):
        children = self.graph.predecessors(label)
        for child in children:
            if 'result' not in self.task_dict[child]:
                return False 
        return True


    def report_status(self):
        charlist = []
        for t in self.tasklist:
            if 'result' not in self.task_dict[t]:
                char = '.'
            else:
                res = self.task_dict[t]['result']
                if res is None:
                    char = 'X'
                else:
                    char = 'C'
            charlist.append(char)
            
        print("".join(charlist))
            
    def draw_graph(self):
        plt.clf()
        g = nx.reverse(self.graph)
        pos = nx.planar_layout(nx.reverse(g))
        nx.draw(g, pos=pos, node_color='pink')
        nx.draw_networkx_labels(g, pos=pos)


class LinearPipeline(Pipeline):
    """A special case of the pipeline where each task depends on, and only on,
    the previous task. Feed this class an ordered list of tasks and it
    will generate the tree without needing to explicitly state the dependencies

    Usage: ::

        t1 = Task1()
        t2 = Task2()
        ...
        tn = TaskN()

        pipeline = LinearPipeline([t1, t2, ... tn])
    """

    def __init__(self, tasks, **kwargs):
        tasklist = [ ('t0', tasks[0])]

        #Convert the list of tasks into format the Pipeline expects
        for i in range(1, len(tasks)):
            tasklist.append( (f't{i}', tasks[i], f't{i-1}') )
        Pipeline.__init__(self, tasklist, **kwargs)


class BranchingPipeline(Pipeline):
    """Take one of two different paths depending on the result of testFunc
    
        THIS IS STILL A SKETCH
    """
    def __init__(self, truePath, falsePath, testFunc=None):

        #Should we inject the test function, or inherit and reimplement?
        # assert isinstance(testFunc, function), "testFunc must be a function"
        if testFunc is not None:
            self.testFunc = testFunc 

        self.truePath = truePath 
        self.falsePath = falsePath  
        self.validate()

    def validate(self):
        self.truePath.validate() and self.falsePath.validate()

        sig1 = self.truePath.get_input_signature()
        sig2 = self.falsePath.get_input_signature()
        if sig1 != sig2:
            raise ValidationError("True and False paths have different input signatures")

        print("Brancing Pipeline Validated")
        # sig1 = self.truePath.get_output_signature()
        # sig2 = self.falsePath.get_output_signature()
        # if sig1 != sig2:
        #     raise ValidationError("True and False paths have different output signatures")


    def get_input_signature(self):
        """See note on chaining pipelines
        """
        return self.truePath.get_input_signature()


    def get_output_signature(self):
        """See note on chaining pipelines
        """
        return self.truePath.get_output_signature()

    def can_depend_on(self, task2):
        return self.truePath.can_depend_on(task2)

    def run(self, *args):
        if self.testFunc(*args):
            print("Running True path")
            return self.truePath.run(*args)
        else:
            print("Running False path")
            return self.falsePath.run(*args)

    def testFunc(self, *args):
        return True 


class ForEachPipeline(Pipeline):
    """Takes a list of inputs and runs each one in turn through
    a pipeline. 

    For example. Here's a pipeline that computes the sum of 
    x**2+1 for a bunch of numbers. Obviously you can do this 
    in one line in numpy, but this example illustrates the usecase::

        class Create(Task):
            def func(self) -> list:
                return list(range(5))

        class SquareTask(Task):
            def func(self, num:int) -> int:
                return num*num 

        class AddOne(Task):
            def func(self, num:int) -> int:
                return num + 1

        class Sum(Task):
            def func(self, values:list) -> np.int64:
                return np.sum(values)

        def foo():
            #Compute sum( x**2 +1) for [0..5)
            t1 = Create(5)
            t2 = LinearPipeline([SquareTask(), AddOne()])
            t2 = ForEachPipeline(t2)

            t3 = Sum()
            p2 = LinearPipeline([t1, t2, t3])


    Todo
    ------
    `get_input_signature` and `get_output_signature` should also tell
    you what the subpipeline is expecting in terms of datatype
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline 

    def run(self, *args):
        feed = args[0]
        assert isinstance(feed, list)

        output = lmap(self.pipeline.run, feed)
        return output

    def validate(self):
        # return self.pipeline.validate()
        return True

    def get_input_signature(self):
        """See note on chaining pipelines
        
        TODO
        The return type should be a list of the type accepted by self.pipeline
        """
        return (list,)

    def get_output_signature(self):
        """See note on chaining pipelines
        TODO
        The return type should be a list of the type return  by self.pipeline
        
        """
        return list

        
# 