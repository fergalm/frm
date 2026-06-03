
try:
    import PyQt6.QtWidgets as QtWidget
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
    import PyQt6.QtWidgets
except ImportError:
    import PyQt5.QtWidgets as QtWidget
    import PyQt5.QtCore as QtCore
    import PyQt5.QtGui as QtGui
    import PyQt5.QtWidgets
    import PyQt5.Qt as Qt


import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


"""
TODO:
o Screen refrsh should remember zoom level
o y value button should be toggled on start
o Cmd line interface
"""


class PlotState:
    def __init__(self, cols):
        self.colNames = list(cols)
        self.xCol = cols[0]
        self.yCols = np.zeros(len(cols), dtype=bool)
        self.yCols[1] = True

        self.xLog = False
        self.yLog = False
        self.showSymbols = True
        self.lineStyle = '-'
        self.zoomLock = False
        self.axl = None    #Axis limits



class QdPlot(QtWidget.QDialog):
    def __init__(self, df, state=None):
        super().__init__(None)
        app = QtWidget.QApplication.instance()
        if app is None:
            app = QtWidget.QApplication([])

        if state is None:
            self.state = PlotState(df.columns)
        else:
            self.state = state
        self.df = df

        self.build()
        self.keyReleaseEvent = self.process_key_press
        #
        self.show()
        self.update()

        #Give plot window a nice aspect
        size = self.size()
        size.setWidth( int(min(1.5*size.height(), 1500)) )
        self.resize(size)

        self.axl = None

    def resizeEvent(self, event):
        pass
        # print(event.size())

    def getState(self):
        return self.state

    def build(self):

        self.control_panel = QtWidget.QVBoxLayout()
        self.setup_control_panel()

        self.plot_panel = QtWidget.QVBoxLayout()
        self.setup_plot_panel()

        layout = QtWidget.QHBoxLayout()
        layout.addLayout(self.control_panel)
        layout.addLayout(self.plot_panel)
        self.setLayout(layout)
        self.setWindowTitle("QdPlot")


    def setup_plot_panel(self):
        panel = self.plot_panel

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        rect = [0.1 ,0.1, .9, .9]
        self.ax = self.canvas.figure.add_axes(rect)

        self.toolbar = NavigationToolbar(self.canvas, self)

        panel.addWidget(self.toolbar)
        panel.addWidget(self.canvas)


    def setup_control_panel(self):
        panel = self.control_panel

        #Create dropdown for xCol
        combo = QtWidget.QComboBox()
        for col in self.state.colNames:
            combo.addItem(col)
        combo.setCurrentText(self.state.xCol)

        panel.addWidget(QtWidget.QLabel("X Column"))
        panel.addWidget(combo)
        combo.currentIndexChanged.connect(self.newXColSelected)
        # combo.show()

        panel.addWidget(QtWidget.QLabel("Line Style"))
        linestyle = QtWidget.QComboBox()
        linestyle.addItem("Solid")
        linestyle.addItem("Dashed")
        linestyle.addItem("Dotted")
        linestyle.addItem("None")

        curIndex = "- -- : None".split().index(self.state.lineStyle)
        linestyle.setCurrentIndex(curIndex)
        linestyle.currentIndexChanged.connect(self.lineStyleSelectSlot)
        panel.addWidget(linestyle)

        button = QtWidget.QPushButton("&Symbols")
        button.clicked.connect(self.showSymbolSlot)
        button.setCheckable(True)
        button.setChecked(True)
        panel.addWidget(button)

        button = QtWidget.QPushButton("Lock &Zoom")
        button.clicked.connect(self.lockZoomSlot)
        button.setCheckable(True)
        button.setChecked(self.state.zoomLock)
        panel.addWidget(button)

        xbutt = QtWidget.QPushButton("&X Log")
        xbutt.setCheckable(True)
        xbutt.setChecked(self.state.xLog)
        xbutt.clicked.connect(self.xLogSlot)

        ybutt = QtWidget.QPushButton("&Y Log")
        ybutt.setCheckable(True)
        ybutt.setChecked(self.state.yLog)
        ybutt.clicked.connect(self.yLogSlot)

        loglayout = QtWidget.QHBoxLayout()
        loglayout.addWidget(xbutt)
        loglayout.addWidget(ybutt)
        panel.addLayout(loglayout)


        panel.addStretch()

        #Create buttons for each column
        group = QtWidget.QGroupBox("Select Columns")

        if True:
            panel.addWidget(group)
        else:
            group.setMinimumSize(150, 800)
            group.setMinimumSize(200, 1600)

            #Something like this will get scrollbars working
            scroll = QtWidget.QScrollArea(self)
            scroll.setWidget(group)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy( QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff )
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded )
            panel.addWidget(scroll)


        group_layout = QtWidget.QVBoxLayout()
        group.setLayout(group_layout)

        for i, col in enumerate(self.state.colNames):
            button = QtWidget.QPushButton(col)
            button.setCheckable(True)

            if self.state.yCols[i]:
                button.setChecked(True)

            button.clicked.connect(self.newYColToggled)
            group_layout.addWidget(button)



    #TODO rename xColSelectSlot
    def newXColSelected(self, index):
        self.state.xCol = self.state.colNames[index]
        self.update()

    def lineStyleSelectSlot(self, val):

        #TODO label names defined in 2 places!
        #opts = {
            #'Solid': "-",
            #'Dashed': '--',
            #'Dotted': ':',
            #"None": None,
        #}
        opts = ["-", "--", ":", "None"]
        self.state.lineStyle = opts[val]
        print(f"Setting linestyle to {self.state.lineStyle}")
        self.update()

    def lockZoomSlot(self):
        self.state.zoomLock = not self.state.zoomLock

    def showSymbolSlot(self):
        self.state.showSymbols = not self.state.showSymbols
        self.update()

    def xLogSlot(self):
        self.state.xLog = not self.state.xLog
        self.update()

    def yLogSlot(self):
        self.state.yLog = not self.state.yLog
        self.update()

    @QtCore.pyqtSlot()
    def newYColToggled(self):
        sender = self.sender()
        name = sender.text()

        index = self.state.colNames.index(name)
        self.state.yCols[index] = not self.state.yCols[index]
        self.update()

    def update(self):
        print("Update called")
        st = self.state

        axl = None
        if self.state.zoomLock:
            axl = self.state.axl
            print(f"Zoom lock is on. {axl=}")


        self.state.axl = qdplot(self.ax, self.df, st.xCol, st.yCols, ls=st.lineStyle, show_symbols=st.showSymbols, xLog=st.xLog, yLog=st.yLog, axl=axl)
        self.canvas.draw_idle()

    def process_key_press(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.hide()
            self.close()



def qdplot(ax, df, xCol, yColBool, ls="-", show_symbols=True, xLog=False, yLog=False, axl=None):
    cols = df.columns


    symbols = "Posv*DXP"
    ax.cla()

    xval = df[xCol]
    for i, flag in enumerate(yColBool):
        if not flag:
            # print(f"Not plotting {i} {cols[i]}")
            continue

        # print(f"Plotting {i} {cols[i]}")
        yval = df[ cols[i] ]

        sym = ","
        if show_symbols:
            sym = i % len(symbols)
            sym = symbols[sym]

        #zeroth call gets 16th colour, 1st col gets zeroth colour
        clr = (i-1) % 16
        ax.plot(xval, yval, f"C{clr}{sym}",
                ls=ls,
                label=cols[i]
        )

    ax.legend()

    if xLog:
        ax.set_xscale('log')

    if yLog:
        ax.set_yscale('log')

    if axl is not None:
        ax.axis(axl)

    return ax.axis()
