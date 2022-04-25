import PyQt5.QtWidgets
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore



class QTable():
    """
    Make a popup interactive table of data for easier viewing.

    Not a plot, per se, but uses the same calls to PyQt as matplotlib
    does, so it lives in the same repository

    Inputs
    ---------
    df
        Dataframe to display
    num
        (int) Max number of rows to display
    numGuides
        (int) Every nth row is drawn with a different background colour
        to guide the eye


    TODO:
    ------
    x Return control to the prompt
    x Better highlight color
    o Show/hide column
    o Change column order?
    x Sort dataframe by column
    x Better sort
    o format strings?
    o max column widths?
    o Better default width and height


    Future Work
    o Select columns and plot.
    """
    def __init__(self, df, num=1000):
        if len(df) > num:
            df = df[:num]

        self.df = df 
        self.nrow = len(df)
        self.ncol = len(df.columns)
        self.app, self.table = self.create()
        
        self.set_size_policy()
        self.draw_row_guides(None)
        print("Press [Q] in window to quit")

    def create(self):
        app = PyQt5.QtWidgets.QApplication.instance()
        if app is None:
            app = PyQt5.QtWidgets.QApplication([])

        app.setTitle = "Pandas DataFrame"
        tab = PyQt5.QtWidgets.QTableWidget(self.nrow, self.ncol)
        tab = self.set_table_elements(tab, self.df)
        # tab.itemClicked.connect(print_which_cell_clicked)

        header = tab.horizontalHeader()
        header.sectionClicked.connect(self.draw_row_guides)
        tab.show()
        tab.setSortingEnabled(True)
        
        return app, tab 

    def set_size_policy(self):
        tab = self.table
        width_pix = tab.horizontalHeader().length() + tab.verticalHeader().width() + 20
        height_pix = tab.verticalHeader().length() + tab.horizontalHeader().width()
        tab.setMaximumSize(width_pix, height_pix)

        width_pix = min(width_pix, 1000)
        height_pix = min(height_pix, 1000)
        tab.resize(width_pix, height_pix)

        #I think I have to subclass QTableWidget and override keyReleaseEvent
        tab.keyReleaseEvent = self.quit

    def quit(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.table.hide()



    def set_table_elements(self, tab, df):
        tab.setHorizontalHeaderLabels(df.columns)

        QItem = PyQt5.QtWidgets.QTableWidgetItem  #Mnumonic
        for i, key in enumerate(df.columns):
            col = df[key]
            for j, elt in enumerate(col):
                try:
                    item = QItem("%g"% (elt))
                except TypeError:
                    item = QItem(str(elt))
                #Mark item as readonly
                item.setFlags( item.flags() & ~QtCore.Qt.EditRole)
                tab.setItem(j, i, item)

        tab.resizeColumnsToContents()
        tab.resizeRowsToContents()
        return tab 

    def draw_row_guides(self, index):
        """This is called by the HeaderView.isClicked signal"""
        guideStep = 5

        for i in range(self.nrow):
            clr = QtGui.QColor('#FFFFFF')
            if i % guideStep == guideStep -1:
                clr = QtGui.QColor('#DDDDFF')

            for j in range(self.ncol):
                item = self.table.item(i, j)
                item.setBackground(clr)


# def display():
#     """Taken from inputhookqt4.py in IPython"""

#     try:
#         allow_CTRL_C()
#         app = QtCore.QCoreApplication.instance()
#         if not app: # shouldn't happen, but safer if it happens anyway...
#             return 0
#         app.processEvents(QtCore.QEventLoop.AllEvents, 300)
#         if not stdin_ready():
#             # Generally a program would run QCoreApplication::exec()
#             # from main() to enter and process the Qt event loop until
#             # quit() or exit() is called and the program terminates.
#             #
#             # For our input hook integration, we need to repeatedly
#             # enter and process the Qt event loop for only a short
#             # amount of time (say 50ms) to ensure that Python stays
#             # responsive to other user inputs.
#             #
#             # A naive approach would be to repeatedly call
#             # QCoreApplication::exec(), using a timer to quit after a
#             # short amount of time. Unfortunately, QCoreApplication
#             # emits an aboutToQuit signal before stopping, which has
#             # the undesirable effect of closing all modal windows.
#             #
#             # To work around this problem, we instead create a
#             # QEventLoop and call QEventLoop::exec(). Other than
#             # setting some state variables which do not seem to be
#             # used anywhere, the only thing QCoreApplication adds is
#             # the aboutToQuit signal which is precisely what we are
#             # trying to avoid.
#             timer = QtCore.QTimer()
#             event_loop = QtCore.QEventLoop()
#             timer.timeout.connect(event_loop.quit)
#             while not stdin_ready():
#                 timer.start(50)
#                 event_loop.exec_()
#                 timer.stop()
#     except KeyboardInterrupt:
#         global got_kbdint, sigint_timer

#         ignore_CTRL_C()
#         got_kbdint = True
#         # mgr.clear_inputhook()


# import signal 
# import select
# import sys 

# def allow_CTRL_C():
#     """Take CTRL+C into account (SIGINT)."""
#     signal.signal(signal.SIGINT, signal.default_int_handler)

# def ignore_CTRL_C():
#     """Ignore CTRL+C (SIGINT)."""
#     signal.signal(signal.SIGINT, signal.SIG_IGN)

# def stdin_ready():
#     """Return True if there's something to read on stdin (posix version)."""
#     infds, _, _ = select.select([sys.stdin],[],[],0)
#     return bool(infds)
