from colorama import Fore
import traceback
import datetime
import inspect
import logging
import os

"""Simpler replacement to python's logging infrastructure

I can never understand how to get Python's logging module to work how I want. For example,
why doesn't this code print anything?::

    import logging
    logging.info("Hello")  #Default log level is WARN, don't expect anything
    logging.basicConfig(level=logging.INFO)
    logging.info("Goodbye")  #Also prints nothing


This simplifies things by

* Handles the non-intuitive requirements of setting up a logging object
* Makes it easy to change log level on the fly
* Only sends log messages to stderr. No multiple handlers
* Fixed format for prefix: Looks something like
    INFO 2021-06-23T13:43:41 predict_peak.py:save_result:38 Your message here


Usage
------------::

    logger = flogger.Logger()
    logger.setLevel(flogger.DEBUG)
    logger.warn("This is a warning...")


The module also creates a singleton object that can be used without initialisation. Changes to verbosity
in this singleton object are felt globally. Usage::

    flogger.log.warn("This is a warning")

Notes
---------
The thread-safeness of this code has not been tested, or thought about too much. That said, it
uses the CPython logger under the hood, so it should work.

TODO
--------------
x Add backtrace method
x Better colours?
o Ensure threadsafeness
o A MultiLogger with different log levels and different outputs
o Avoid the duplication inherent in get_level_str()
"""

ERROR = 10
WARN = 20
INFO = 30
DEBUG = 40


class BaseStyle:
    """Defines the API for a stylist object

    A stylist constructs a string to be logged based on the input level and messagge.
    It is similar in intent to the Formatter class in the Cpython logging module, but
    given a different name to avoid confusion. The API is quite different
    """

    def __call__(self, level, msg):
        return self.format_msg(level, msg)

    def format_msg(self, level, msg):
        """Reimplement this in the daughter classes"""
        return msg


class DefaultStyle(BaseStyle):
    def format_msg(self, level, msg):
        date_fmt = "%Y-%m-%dT%H:%M:%S"

        level_str, level_clr = self.get_level_str(level)
        location_str = self.get_location_str()
        date = datetime.datetime.now().strftime(date_fmt)
        output = level_clr + "%s %s %s " % (level_str, date, location_str)
        output = output + Fore.RESET + msg
        return output

    def get_level_str(self, level):
        names = {
            10: ("ERROR", Fore.RED),
            20: ("WARN", Fore.MAGENTA),
            30: ("INFO", Fore.CYAN),
            40: ("DEBUG", Fore.RESET),
        }
        return names.get(level, "L%02i" % (level))

    def get_location_str(self):
        # Get the calling frame
        frame = inspect.currentframe()
        for i in range(5):
            frame = frame.f_back

        (filename, lineno, funcname, _, _) = inspect.getframeinfo(frame)
        filename = filename.split(os.path.sep)[-1]
        return "%s:%s:%s" % (filename, funcname, lineno)


class MonochromeStyle(DefaultStyle):
    """Prints everything in black and white"""

    def get_level_str(self, level):
        names = {
            10: ("ERROR", ""),
            20: ("WARN", ""),
            30: ("INFO", ""),
            40: ("DEBUG", ""),
        }
        return names.get(level, "L%02i" % (level))


class Logger:
    def __init__(self, level=INFO, style=DefaultStyle()):
        # Sometimes I pass a class instead of an object. Intent is the same
        if isinstance(style, type):
            style = style()

        self.stylist = style
        self.level = level

        # Create the python logger
        self.log = logging.Logger("Flogger")
        if len(self.log.handlers) == 0:
            format = logging.Formatter("%(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(format)
            self.log.addHandler(handler)
        assert len(self.log.handlers) == 1

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

    def log_exception(self, exc, level=ERROR):
        """Log an exception and its backtrace as an error message

        Example::

            try:
                1/0
            except ZeroDivisionError as e:
                log.log_exception(e)
        """
        msg = str(exc)
        msg = msg + "\n" + traceback.format_exc()
        self.write(level, msg)

    def write(self, level, msg):
        if level > self.level:
            return
        msg = self.stylist(level, msg)
        self.log.log(1, msg)


# Call this object directly if you don't want to muck about with creating multiple log objects
log = Logger()
