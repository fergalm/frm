
"""
Sketches of a pipeline and task pair for abstracting
out the running of a sequency of steps from the 
business logic.

Lots of stuff from the old clips and tasks code 
would be appropriate in the Pipelines.

The BaseTask has a neat way of checking that it is
compatible with adjacent pipeline steps before being
run
"""

class BaseTask():
    def __init__(self, dtypein, dtypeout, config):
        self.dtypein = dtypeout
        self.dtypeout = dtypeout
        self.config = config
        
    def validate(self, data, dtype):
        #TODO. I need a validator class that
        #can check things like does the dataframe
        #have the correct columns in it
        assert isinstance(data, dtype)
        
    def run(self, *args):
        """This is the method called by the pipeline"""
        self.validate(*args, self.dtypein)
        value = self.foo(*args)
        self.validate(value, self.dtypeout)
        return value

    def process(self):
        raise NotImplementedError("Put your logic in this function in the daughter class")

        
class SparkSessionTask(BaseTask):
    def __init__(...)
        self.session = spark.new_session()
    
class ExtractTask(SparkSessionTask):
    def __init__(self, session, config)
        self.validateConfig(config)
        self.SparkSessionTask.__init__(None, pd.DataFrame, config))
        
    def self.validateConfig(self, config):
        assert "uname pwd".split() in config
        
    def process(self, *args):
        extract_function(self.session, *args)
        
        
class TransformTask(BaskTask):
    ...
        
        
class LinearPipeline():
    def __init__(self, steps):
        self.steps = steps 
        
    def validate(self):
        for i in range(len(self.steps)-1):
            step1 = self.step[i]
            step2 = self.step[i+1]
            
            assert self.iscompatible(step1.__output__, step2.__input__)
        
    def execute(self):
        val = None
        for step in self.steps:
            val = step(val)
        return val
        
        
class TreePipeline():
    """Able to process pipelines were some tasks
    depend on >1 output, or >1 tasks depend on a preceeding
     task
     
     A ----- C ------D
     B ---/     \___E
     
     Lots of logic in the graph object not figured out yet
     """
     
    def execute():
        queue = self.figure_out_order_of_operations(self.graph)
        for task in queue:
            deps = self.graph.dependencies(task)
            value =  task.run(deps)
            self.graph[task] = value 
            self.garbage_collect(task)
            
        return value
            
            
    
def main():
    session = spark.new_session()
    task1 = ExtractTask(session)
    task2 = TransformTask()
    
    pipeline = TreePipeline(
        ['extract', task1, []],
        ['transform', task2, ['task1']
    )
    
    pipeline.validate()
    pipeline.run()
    
    
