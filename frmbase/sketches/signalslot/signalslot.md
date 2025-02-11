% Signals and Slots (The observer pattern)
% Fergal Mullally
% 2024-10-08



## Motivation

Feature Request:

> Tara wants to be sent an email whenever the hydro scheduler uses the spillway.

(This is hypothetical)

## The code

Currently, our `pipeline.py` looks something like

```python
class Pipeline:

    def do():
        modelList = [SolveNoSpillway(), SolveWithSpillway(), ...]
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue
```

* We first try to solve with the spillway turned off

* If that fails, try again with the spillway turned on

  



## First Attempt

```python
from email import send_an_email

class Pipeline:
    def do():
        modelList [SolveNoSpillway(), SolveWithSpillway(), ...]
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue

            # This is new
            if model.spillway is True:
                send_an_email(...)
```


We don't like this because for three reasons.

* `pipelines.py` now has a new dependency ☹

* `do()` is doing two things ☹

  * Trying different models, 
  * Sending emails

* Half our unit tests will spam us all the time ☹

  

## Second Attempt
(Dependency Inversion)

```python

class Pipeline:
    def __init__(self, emailer:AbstractEmailer):
        ...

    def do():
        modelList [SolveNoSpillway(), SolveWithSpillway(), ...]
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue

            if model.spillway is True:
                self.emailer.send_an_email(...) # This is new
```


+ Duck typing 

  + As long as `emailer` has a `send_an_email` method, all will be well.

+ No **explicit** dependency on emailer (i.e no import statement) ✅
+ Easy to substitute in a DummyEmailer for test. ✅

## But...
Still doesn't feel right.

* What if we have to do other kinds of reporting in the future. Initialisation in future might require passing in bunches of abstract classes. ☹

*  `do()` still does two things ☹



## Third Attempt

```python

class Pipeline:
    def __init__(self):
		#This is new 
        self.messenger = Signal()

    def do():
        modelList [SolveNoSpillway(), SolveWithSpillway(), ...]
        for model in modelList:
            try:
                model.setup_and_solve()
            except SolveFailedError:
                continue
                
            # This is new
            self.messenger.emit(f"Spillway is {model.spillway}")

```





## Third Attempt (Part 2)

```python
def main()
    pipeline = Pipeline()
    pipeline.messenger.connect(receive)

    pipeline.do()

    
def receive(message):
    if "True" in message:
        send_an_email()
    
```



## Third Attempt (Part 3)

* Pipelines has no external dependencies ✅

* Clean Code: Pipelines.do() only worries about doing one job ✅

* Future Proof: If the requirements of what we want to do when spillway is used changes, the pipeline code will remain unchanged. ✅

* Unit testing is easy. Simply leave the signal unconnected so nothing listens to it. ✅

  



## Signals

Signals are like mini exceptions. They shout a message to the world, then carry on execution. They don't care if the message is heard, that's not their concern.



```python
is_bad = somefunction()
if is_bad:
	signal.emit()    #Alert and continue 
	raise Exception  #Alert and stop work

```





## Signal Internals

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



.



.

```python

class Foo:
    def __init__(self):
        self.signal = Signal()
        
foo = Foo()
foo.signal.connect(a_function_that_knows_what_to_do)
```



## Names and Alias

* A function/class that shouts into the void is called a **Signal**. A function that is called by a signal is called a **Slot**
  * These names mimic those used by Qt software for user interfaces



Many other names exist

* Listeners
* Pub/Sub (Publish/Subscribe architecture)
* Observer Pattern



## Design Philosophy

Signals/Slots are a specific example of a broader idea in software design, the onion architecture.



1. Software should be designed like the layers of an onion

2. Internal layers don't know about external layers

3. External layers know only the interface to next internal layer

   1. The Business Logic should not know anything about platform it runs on
   2. The code that runs the business logic must know about the logic, but limited exposure to hardware (i.e it knows it runs on a website, but doesn't know it's an Azure website
   3. Outermost layer knows all the details of the hardware, limited information on the business logic.

    

![](/home/fergal/all/frm/frmbase/sketches/signalslot/onion.png)

## Questions?



## Bonus: Class and object signals 

```python
   class Worker:
        #Listeners to this signal listen to every object
        class_signal = Signal()

        def __init__(self, name):
            self.name = name
            #Listeners to this signal only listen to a single object
            self.object_signal = Signal()

        def apply(self):
            self.class_signal.emit(self.name)
            self.object_signal.emit(self.name)

```



## Class and object signals (2)

```python
def slot1(caller):
        print(f"Slot 1 called by {caller}")

def slot2(caller):
    print(f"Slot 2 called by {caller}")

#Connect slot1 to all workers
Worker.class_signal.connect(slot1)

w1, w2 = Worker("W1"), Worker("W2")

#Connect slot two to a single worker
w1.object_signal.connect(slot2)


```



```python
w1.apply()
>>> Slot 1 called by W1
>>> Slot 2 called by W1

w2.apply()
>>> Slot 1 called by W2

```



## Bonus: Type Checking



```python
class Foo
	def __init__(self):
		#Specify the argument types of the slots
		self.signal = Signal( (int, str) )
		
def listener(message:str):
	... 
	
def main():
	f = Foo()
	f.signal.connect(listener)  #raises an exception
```



* Runtime type checking in Python is really undercooked
* Type hints must match exactly
  * `list != typing.List` 
  * `List != List[str]`
  * `int != typing.Any`
* `emit` is still defined as `emit(*args, **kwargs)`
  * Developer must ensure `emit` is called with the promised args
  * But type signature and call to `emit` are in the same class, so hopefully that's easier.



## Bonus: Connecting signals 



Signals can be chained so they propagate up the calling stack. Unlike exceptions, this must

be done explicitly.

```python
class Under:
    under_signal = Signal()
    
class Over:
	def __init__(self):
        self.over_signal = Signal()
        self.under = Under()
        self.under.under_signal.connect(self.over_signal.emit)
        
def slot():
    print("signal heard")
    
def main():
    over = Over()
    over.over_signal.connect(slot)
    
    over.under.under_signal.emit() # Slot is called
```

