
"""
Sketch of an algorithm for working in an event driven architecture

signal.py is probably a better approach
"""


def controller():
    
    handler_dict = loadHandlers()
    
    while True:
        while len(eventList) > 0:
            event = eventList.pop(0)

            handler = handler_dict.get(event, handle_unknown_event)
            new_events = handler(event)
            eventList.extend(new_events)
            
        eventList = getExternalEvents()
        
        
        
def loadHandlers():
    out = {
        LoggingEvent: handle_logging_event,
        OutOfMemoryEvent: handle_oom,
        OutOfSpaceEvent: handle_oos
    }
    return out
    

    
    
class Event:
    pass 
    
class LoggingEvent(Event):
    def __init__(self, msg):
        self.msg = msg 
        
class OutOfMemoryEvent(Event):
    ...
    

class OutOfSpaceEvent(Event):
    ... 

class EmailEvent(Event):
    def __init__(self, to, subject, msg):
        ... 


def handle_logging_event(event):
    msg = event.msg 
    print("LOG: %s" %(msg))


def handle_oom(event):
    #Garbage collect
    ...

    return [LoggingEvent("Freed up some memory")]


def handle_oos(event):
    #Free some temporary files

    events = [
        LoggingEvent("Freed up some memory"),
        EmailEvent("admin@example.com", "We need more disk space"),
    ]
    return events

def handle_unknown_event(event):
    ...
