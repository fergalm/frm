## Motivation

Feature Request:

> Tara wants to be sent an email whenever the hydro scheduler uses the spillway.

(This is hypothetical)

## The code

Our `pipeline.py` looks something like

```python
class Pipeline:

    def do():
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue
```



## First Attempt

```python
from email import send_an_email

class Pipeline:
    def do():
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue

            # This is new
            if model.spillway is True:
                send_an_email(...)
```


We don't like this because for two reasons.

* `pipelines.py` now has a new dependency :frowning_face:

* Half our unit tests will spam us all the time :frowning_face:

  

## Second Attempt
(Dependency Inversion)

```python

class Pipeline:
    def __init__(self, emailer:AbstractEmailer):
        ...

    def do():
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue

            if model.spillway is True:
                self.emailer.send_an_email(...) # This is new
```


+ Depending on an abstraction. :white_check_mark:
+ Easy to substitute in a DummyEmailer for test. :white_check_mark:

But...
Still doesn't feel right.
What if we have to do other kinds of reporting in the future. Initialisation in future might require passing in bunches of abstract classes. :frowning_face:

More nagging, `do()` function is nearly twice as long. And it's spending lines of code on something nothing to do with pipelines. :frowning_face:




## Third Attempt
```python

class Pipeline:
    def __init__(self, emailer:AbstractEmailer):
		#This is new 
        self.messenger = Signal()

    def do():
        for model in modelList:
            try:
                model.setup_and_solve()
                # This is new
                self.messenger.emit(f"Spillway is {model.spillway}")
            except SolveFailedError:
                continue



class Monitor:
    def receive(self, message):
        if "True" in message:
            send_an_email()

def main()
    monitor = Monitor()
    pipeline = Pipeline()
    pipeline.messenger.connect(monitor.receive)

    pipeline.do()
```



* Pipelines has no external dependencies :white_check_mark:

* Pipelines worries only about running each model in sequence, leaves details of what to do when things go wrong to someone else :white_check_mark:

* Unit testing is easy. Simply leave the signal unconnected so nothing listens to it. :white_check_mark:

* If the requirements of what we want to do when spillway is used changes, the pipeline code will remain unchanged. :white_check_mark:

  

Signals are like mini exceptions. They shout a message to the world, then carry on execution. They don't care if the message is heard, that's not their concern.



```python
is_bad = somefunction()
if is_bad:
	signal.emit()    #Alert and continue 
	raise Exception  #Alert and stop work

```






### Signal Internals

Signals are really simple objects
```python
class Signal:
    def __init__(self):
        self.listeners = []

    def connect(self, func):
        self.listeners.append(func)

    def emit(self, *args, **kwargs):
        for func in self.listeners:
            func(*args, **kwargs)
```



### Type Checking

Can we type check the signal and the slot?
Can we check somewhere along the line that the slot can accept what the signal emits?

Option 1: `def emit(self):   #No args`
Option 2: `def emit(self, arg: AbstractEvent):   #Fixed type`
Option 3: `class CustomSignal: def emit(self, msg:str):  #Custom signals`
