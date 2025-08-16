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

import matplotlib.pyplot as plt
from pandas._libs.tslibs.parsing import DateParseError
import pandas as pd
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

class QdPlot(QtWidget.QDialog):
    def __init__(self, df):
        super().__init__(None)
        app = QtWidget.QApplication.instance()
        if app is None:
            app = QtWidget.QApplication([])

        self.df = df
        self.columns = list(df.columns)
        self.xcol = self.columns[0]
        self.ycols = np.zeros(len(self.columns))
        self.ycols[1] = True

        self.linestyle = "-"
        self.show_symbols = True
        self.xLog = False
        self.yLog = False

        self.build()
        self.keyReleaseEvent = self.process_key_press

        self.show()
        self.update()


    def build(self):
        self.setWindowTitle("QdPlot")
        self.layout = QtWidget.QHBoxLayout()
        self.control_panel = QtWidget.QVBoxLayout()
        self.control_panel.maximumWidth = 200

        self.plot_panel = QtWidget.QVBoxLayout()
        self.layout.addLayout(self.control_panel)
        self.setup_control_panel()

        self.layout.addLayout(self.plot_panel)
        self.setup_plot_panel()

        self.setLayout(self.layout)

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
        for col in self.columns:
            combo.addItem(col)

        panel.addWidget(QtWidget.QLabel("X Column"))
        panel.addWidget(combo)
        combo.currentIndexChanged.connect(self.newXColSelected)
        combo.show()

        panel.addWidget(QtWidget.QLabel("Line Style"))
        linestyle = QtWidget.QComboBox()
        linestyle.addItem("Solid")
        linestyle.addItem("Dashed")
        linestyle.addItem("Dotted")
        linestyle.addItem("None")
        linestyle.currentIndexChanged.connect(self.lineStyleSelectSlot)
        panel.addWidget(linestyle)

        button = QtWidget.QPushButton("&Symbols")
        button.clicked.connect(self.showSymbolSlot)
        button.setCheckable(True)
        button.setChecked(True)
        panel.addWidget(button)


        xbutt = QtWidget.QPushButton("&X Log")
        xbutt.setCheckable(True)
        xbutt.clicked.connect(self.xLogSlot)

        ybutt = QtWidget.QPushButton("&Y Log")
        ybutt.setCheckable(True)
        ybutt.clicked.connect(self.yLogSlot)

        loglayout = QtWidget.QHBoxLayout()
        loglayout.addWidget(xbutt)
        loglayout.addWidget(ybutt)
        panel.addLayout(loglayout)


        panel.addStretch()

        #Create buttons for each column
        group = QtWidget.QGroupBox("Select Columns")
        panel.addWidget(group)

        group_layout = QtWidget.QVBoxLayout()
        group.setLayout(group_layout)

        for i, col in enumerate(self.columns):
            button = QtWidget.QPushButton(col)
            button.setCheckable(True)

            if i == 1:
                button.setChecked(True)

            button.clicked.connect(self.newYColToggled)
            group_layout.addWidget(button)

        ##Press one of the buttons
        #child = group_layout.findChild(
            #QtWidget.QPushButton,
            #self.df.columns[1]
        #)
        #child.setChecked(True)


    #TODO rename xColSelectSlot
    def newXColSelected(self, index):
        self.xcol = self.columns[index]
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
        self.linestyle = opts[val]
        print(f"Setting linestyle to {self.linestyle}")
        self.update()

    def showSymbolSlot(self):
        self.show_symbols = not self.show_symbols
        print(f"Show Symbol: {self.show_symbols}")
        self.update()

    def xLogSlot(self):
        self.xLog = not self.xLog
        self.update()

    def yLogSlot(self):
        self.yLog = not self.yLog
        self.update()

    @QtCore.pyqtSlot()
    def newYColToggled(self):
        sender = self.sender()
        name = sender.text()
        index = self.columns.index(name)

        self.ycols[index] = not self.ycols[index]
        self.update()

    def update(self):
        print("Update called")
        qdplot(self.ax, self.df, self.xcol, self.ycols, ls=self.linestyle, show_symbols=self.show_symbols, xLog=self.xLog, yLog=self.yLog)
        self.canvas.draw_idle()

    def process_key_press(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.hide()
            self.close()


def process(df):

    xcol= None
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], utc=True)

            if xcol is None:
                xcol = col
        except DateParseError:
            pass

        try:
            #Can we cast as a float?
            df[col].astype(float)

            if xcol is None:
                xcol = col
        except (TypeError, ValueError):
            pass

    return df, xcol


def qdplot(ax, df, xCol, yColBool, ls="-", show_symbols=True, xLog=False, yLog=False):
    cols = df.columns

    symbols = "Posv*DXP"
    ax.cla()
    xval = df[xCol]
    for i, flag in enumerate(yColBool):
        if not flag:
            print(f"Not plotting {i} {cols[i]}")
            continue

        print(f"Plotting {i} {cols[i]}")
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
