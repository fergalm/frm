from ipdb import set_trace as idebug
from frmbase.support import lmap
import matplotlib.pyplot as plt 
import networkx as nx

from task import Task, ValidationError

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
            for r in reqs:
                t1 = self.task_dict[r]['func']
                val &= t0.can_depend_on(t1)

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

