"""
When an error occurs, code throws an exception and halts execution.
Responsibility of dealing with the error is passed to the parent
function.

```python
if something_really_bad_happened:
    #Halt further execution and let calling code deal with it.
    raise ValueError()

this_function_is_not_executed()
```

Signals offer a way to issue a warning and continue execution.

```python

if something_unusual_happened:
    #Tell calling code, and let them deal with it, but otherwise continue
    Signal().emit()

this_function_is_always_executed()
```

Any code that you want to run in response to this warning merely has
to subscribe to the signal so it gets told to run.

### Example usage

```python
class Pipeline:
    def __init__(self, emailer:AbstractEmailer):
		#Create a
        self.messenger = Signal()

    def run():
        for model in modelList:
            try:
                model.setup_and_solve()
                #Signal emits a message
                self.messenger.emit(f"Model {model} converged")
            except SolveFailedError:
                continue

class Monitor:
    def receive(self, message):
        #This function will receive the message
        #we call this the "slot"
        if "LastGaspBackupModel" in message:
            send_an_email()

def main():
    monitor = Monitor()
    pipeline = Pipeline()
    #The monitor slot is connected to the pipeline's signal
    pipeline.messenger.connect(monitor.receive)

    pipeline.do()
```

Signals are not restricted to issuing warnings. They can be used
anywhere a function doesn't want to concern itself with dealing
with an event. The Qt GUI framework uses signals and slots extensively
to respond to mouse and keyboard events. The code for their "OK" button
issues a signal anytime the button is clicked. The button class only
concerns itself with whether it has been clicked or not, and leaves
responsibility for dealing with that event to a listener.
"""

from typing import Callable

class Signal:
    """
    A signal is like an exception which doesn't halt
    code execution.

    It is useful to allow a piece of code encounters
    an event it doesn't know how to deal with. The
    signal tells the calling code that something has happened
    but then carries on.

    A fuller description is in the module level documentation
    """

    def __init__(self):
        self.listeners = []

    def connect(self, func: Callable):
        """Add a function to listens to this signal.

        When the signal emits, every listening function
        will get called in sequence
        """
        self.listeners.append(func)

    def emit(self, *args, **kwargs):
        """Call all the listeners and tell them an event happened

        Inputs
        ----------
        All arguments are passed to the listeners

        Returns
        -----------
        **None**
        """
        for func in self.listeners:
            func(**kwargs)


