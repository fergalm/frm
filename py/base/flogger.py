from ipdb import set_trace as idebug
from colorama import Fore
import traceback
import datetime 
import inspect 
import os 


ERROR=10
WARN=20
INFO = 30
DEBUG = 40

"""
Simpler replacement to python's logging infrastructure

Usage
------------
```
logger = flogger.Logger()
logger.setLevel(flogger.DEBUG)
logger.warn("This is a warning...")
```

The module creates a singleton object that can be used without initialisation. Changes to verbosity
in this singleton object are felt globally: `flogger.log.warn("This is a warning...")`

Notes
---------
The thread-safeness of this code has not been tested, or thought about too much.

TODO
--------------
x Add backtrace method
o Better colours?
o Ensure threadsafeness
o A MultiLogger with different log levels and different outputs
o Avoid the duplication inherent in get_level_str()
"""
class Logger():
    """Simpler replacement to python's logging infrastructure
    
    I can never understand how to get Python's logging module to work how I want.
    This simplifies things by 

    * Making it easy to change log level on the fly
    * Only sends log messages to stdout
    * Fixed format for prefix: Looks something like 
        INFO 2021-06-23T13:43:41 predict_peak.py:save_result:38 Your message here

    """
    def __init__(self, level=DEBUG):
        self.level = level
        self.date_fmt = "%Y-%m-%dT%H:%M:%S"

    def debug(self, msg):
        self.write(DEBUG, msg)

    def info(self, msg):
        self.write(INFO, msg)

    def warning(self, msg):   #Mneumonic for warn
        return self.warn(msg)  
    def warn(self, msg):
        self.write(WARN, msg) 

    def error(self, msg):
        self.write(ERROR, msg) 

    def setLevel(self, level):
        self.level = level 

    def getLevel(self, level):
        return self.level 

    def log_backtrace(self, exc):
        """Log an exception and its backtrace as an error message"""
        msg = str(exc)
        msg = msg + "\n" + traceback.format_exc()
        self.write(ERROR, msg)


    def write(self, level, msg):
        if level > self.level:
            return
        msg = self.format_msg(level, msg)
        print(msg)

    def format_msg(self, level, msg):
        level_str = self.get_level_str(level)
        location_str = self.get_location_str()
        date = datetime.datetime.now().strftime(self.date_fmt)
        clr = self.get_colour(level)
        output = clr + "%s %s %s " %(level_str, date, location_str)
        output = output + Fore.RESET + msg
        return output

    def get_level_str(self, level):
        names = {
            10 : 'ERROR',
            20 : 'WARN',
            30 : 'INFO',
            40 : 'DEBUG', 
        }
        return names.get(level, "L%02i" %(level))

    def get_location_str(self):
        #Get the calling frame
        frame = inspect.currentframe()
        for i in range(4):
            frame = frame.f_back 

        (filename, lineno, funcname, _, _) = inspect.getframeinfo(frame)
        filename = filename.split(os.path.sep)[-1]
        return "%s:%s:%s" %(filename, funcname, lineno)

    def get_colour(self, level):
        if level <= ERROR:
            return Fore.RED
        elif level <= WARN:
            return Fore.YELLOW
        elif level <= INFO:
            return Fore.CYAN
        else:
            return Fore.WHITE
#Call this object directly if you don't want to muck about with creating multiple log objects
log = Logger()

