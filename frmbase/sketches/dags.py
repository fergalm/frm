from ipdb import set_trace as idebug
import matplotlib.pyplot as plt 
from pprint import pprint 

from frmbase.support import lmap
import networkx as nx

"""
Every pipeline should be a task, in that it implements Pipeline.run()

Every task should validate its inputs and outputs at run time
Every task should implmeent validation at compile time.
Every pipeline should validate too. 

For validation to be useful, I must be able to validate columns
(and optionally dtypes!) of dataframes, and for the presence
of keys in dicts
"""

class Task():
    def __call__(self, *args):
        return self.run(*args)

    def run(self, *args):
        print("Running %s with args %s" %(self.name(), args)) 
        return self.name()

    def name(self):
        name = str(self).split()[0][1:]
        return name 

class Extract1(Task):
    ... 

class Extract2(Task):
    ... 

class Transform1(Task):
    ... 

class Transform2(Task):
    ... 

class Merge(Task):
    ... 

class Submit(Task):
    ... 


from networkx.drawing.nx_agraph import graphviz_layout

class Pipeline():
    def __init__(self, tasks):
        self.graph, self.task_dict = self.create_graph(tasks)

        assert nx.is_directed_acyclic_graph(self.graph)
        self.tasklist = list(nx.topological_sort(self.graph))[::-1]
        #print(list(self.tasklist))

    def validate(self):
        pass 

    def run(self):
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

        I don't know how to do this.

        One option is to ask nx for all the immediate descendents
        of a node. If results is set for each descendent, we can
        set results of the tested node to empty. 

        Another option is to create a reverse graph, then
        the edges can be used to 

            Look at digraph.predessors
    "   """
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
        """TODO, there's an easier way to do this"""
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
        pos = nx.planar_layout(self.graph)
        # pos = graphviz_layout(self.graph, prog="dot")
        nx.draw(self.graph, pos=pos, node_color='pink')
        nx.draw_networkx_labels(self.graph, pos=pos)


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
