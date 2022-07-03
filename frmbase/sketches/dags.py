from ipdb import set_trace as idebug
import matplotlib.pyplot as plt 
from pprint import pprint 
import pandas as pd

from frmbase.support import lmap
from inspect import signature
import networkx as nx

from typing import Any
"""
Every pipeline should be a task, in that it implements Pipeline.run()

Every task should validate its inputs and outputs at run time
Every task should implmeent validation at compile time.
Every pipeline should validate too. 

For validation to be useful, I must be able to validate columns
(and optionally dtypes!) of dataframes, and for the presence
of keys in dicts
"""

class ValidationError(Exception):
    pass 

class Task():

    def __call__(self, *args):
        return self.run(*args)

    def get_input_signature(self):
        annotation_list = [x.annotation for x in signature(self.func).parameters.values()]
        return annotation_list        

    def get_output_signature(self):
        try:
            return self.func.__annotations__['return']
        except KeyError:
            return Any
        
    def run(self, *args):
        print("Running %s with args %s" %(self.name(), args)) 
        self.validate_args(args, self.get_input_signature())
        result = self.func(*args)
        self.validate_args(result, self.get_output_signature())
        return result 

    def validate_args(self, actual, expected):
        if not isinstance(actual, tuple):
            actual = [actual]

        if not isinstance(expected, list):
            expected = [expected]

        if len(actual) != len(expected):
            raise ValidationError(f"Expected {len(expected)} arguments, got {len(actual)}")

        i = 0
        for act, exp in zip(actual, expected):
            if exp is Any:
                continue 

            if not isinstance(act, exp):
                raise ValidationError(f"Argument {i}: Expected {exp}, found {act}")
            i += 1

    def can_depend_on(self, task2):
        #I worry this is brittle
        sig1 = self.get_input_signature()
        sig2 = task2.get_output_signature()
        if not isinstance(sig2, list):
            sig2 = [sig2]

        for a, b in zip(sig1, sig2):
            if a != b:
                msg = f"Task {self} expects {sig1} but task {task2} supplies {sig2}"
                print(msg)
                return False 
        return True 
        
    def func(self, df: pd.DataFrame) -> str:
        """Overwrite this function with task logic"""
        return self.name()

    def name(self):
        name = str(self).split()[0][1:]
        return name 

class Extract1(Task):
    __outputs__ = str

class Extract2(Task):
    __outputs__ = str
    ... 

class Transform1(Task):
    __inputs__ = str
    __outputs__ = str

class Transform2(Task):
    __inputs__ = str
    __outputs__ = str

class Merge(Task):
    __inputs__ = (str, str)
    __outputs__ = str

class Submit(Task):
    __inputs__ = str


class DummyTask(Task):
    """A dummy task returns its configuration.

    This is useful when writing pipelines that jump started with some initial values
    """
    def __init__(self, *inputs):
        self.inputs = inputs 

    def func(self):
        return self.inputs


class Pipeline(Task):
    """
        
        Note on Chaining Pipelines
        ---------------------------

        The Pipeline class implements extends the Task class, with 
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

        assert nx.is_directed_acyclic_graph(self.graph)
        assert len(nx.connected_components(self.graph)) == 1, "Islands found in the graph"
        self.tasklist = list(nx.topological_sort(self.graph))[::-1]
        #print(list(self.tasklist))

    def validate(self):
        tasklist = self.tasklist[::-1]

        val = True 
        for label in tasklist:
            t0 = self.task_dict[label]
            reqs = self.graph.successors(label)
            for r in reqs:
                t1 = self.task_dict[r]
                val &= t0.can_depend_on(t1)

        if not val:
            raise ValidationError("Validation failed")

    def get_input_signature(self):
        """See note on chaining pipelines"""
        return self.tasklist[0].get_input_signature()

    def get_output_signature(self):
        """See note on chaining pipelines"""
        return self.tasklist[-1].get_output_signature()

    def run(self):

        #@TODO: Need to pass inputs to the first task
        self.validate(*inputs)

        #Add a dummy task to the task dict and to the task list
        self.task_dict['__dummy_task__'] = dict(func=DummyTask(*inputs), reqs=[])
        self.task_dict[ self.tasklist[0]]['reqs'] = ['__dummy_task__']
        self.tasklist.insert( '__dummy_task__', 0)

        for label in self.tasklist:
            func = self.task_dict[label]['func']
            reqs = self.task_dict[label]['reqs']

            f = lambda x: self.task_dict[x]['result']
            inputs = lmap(f, reqs)

            result = func(*inputs)
            self.task_dict[label]['result'] = result 
            self.garbage_collect(label)
            self.report_status()
        return result

       
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

    def create_graph(self, tasklist):
        labels = dict()
        g = nx.DiGraph()

        for i, row in enumerate(tasklist):
            label = row[0]
            func = row[1]
            deps = row[2:]

            if label in labels:
                raise KeyError
            
            labels[label] = dict(func=func, reqs=row[2:])
            g.add_node(label)
            for d in deps:
                g.add_edge(label, d)
        return g, labels

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
        # pos = graphviz_layout(self.graph, prog="dot")
        nx.draw(g, pos=pos, node_color='pink')
        nx.draw_networkx_labels(g, pos=pos)


class LinearPipeline(Pipeline):
    """A special case of the pipeline where each task depends on, and only on,
    the previous task
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
        if testFunc is not None:
            self.testFunc = testFunc 

        self.truePath = truePath 
        self.falsePath = falsePath

    def validate(self):
        #Validate the true Path 
        #validate the false Path
        ... 

    def run(self, *args):
        if self.testFunc(*args):
            self.truePath.run(*args)
        else:
            self.falsePath.run(*args)

    def testFunc(self, *args):
        return True 


class ForEachPipeline(Pipeline):
    """Takes a list of inputs and runs each one in turn through
    a pipeline. 

    For example. Your previous task might find n canddidate events 
    in a datafeed, and you want to run some analysis on each
    candidate.::

        task1 = FindEventsTask()
        task2 = SpawningPipeline(AnalysisTask())
        task3 = ConcatTask()

        pipeline = LinearPipeline([task1, task2, task3])

    THIS IS STILL A SKETCH
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline 

    def run(self, *args):
        feed = args[0]
        assert isinstance(feed, list)

        output = map(self.pipeline.run, feed)
        return output

def main():

    graph = [
        ('e1', Extract1()),
        ('e2', Extract2()),
        ('t1', Transform1(), 'e1'),
        ('t2', Transform2(), 'e2'),
        ('m1', Merge(), 't1', 't2'),
        ('s1', Submit(), 'm1'),
        ('s2', Submit(), 'm1'),

    ]

    pipeline = Pipeline(graph)
    pipeline.draw_graph()
    pipeline.validate()
    df = pipeline.run()
