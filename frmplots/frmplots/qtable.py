from ipdb import set_trace as idebug 

import PyQt5.QtWidgets as QtWidget
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets



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
    x Better default width and height
    x Press [Q] to quit
    x Sort dataframe by column
    x Better sort
    x Show/hide column
    o format strings?
    o max column widths?
    o A title
    o A button to show the column selector
    o A better class name

    Future Work
    o Select columns and plot.
    o Change column order?
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

        self.selector = ColumnSelector(self)

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
            self.selector.hide()

    def set_table_elements(self, tab, df):
        tab.setHorizontalHeaderLabels(df.columns)

        QItem = PyQt5.QtWidgets.QTableWidgetItem  #Mnumonic
        for i, key in enumerate(df.columns):
            col = df[key]
            for j, elt in enumerate(col):
                if elt is None:
                    item = QItem(" ")
                    continue
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
                if item is not None:
                    item.setBackground(clr)

    def toggleColumn(self, sender_label, state):
        cols = self.df.columns
        for i in range(self.ncol):
            if cols[i] == sender_label:
                if state:
                    self.table.showColumn(i)
                else:
                    self.table.hideColumn(i)

    def showAll(self):
        for i in range(self.ncol):
            self.table.showColumn(i)

    def hideAll(self):
        for i in range(self.ncol):
            self.table.hideColumn(i)


class ColumnSelector(PyQt5.QtWidgets.QDialog):
    def __init__(self, table, parent=None):
        self.table = table
        df = table.df
        PyQt5.QtWidgets.QDialog.__init__(self, parent)
        self.setMinimumWidth(150)

        layout = PyQt5.QtWidgets.QVBoxLayout(self)
        button1 = QtWidget.QPushButton("Select All")
        button1.clicked.connect(self.showAll)
        button2 = QtWidget.QPushButton("Hide All")
        button2.clicked.connect(self.hideAll)
        layout.addWidget(button1)
        layout.addWidget(button2)

        self.boxes = []
        for i, col in enumerate(df.columns):
            print(i, col)
            checkbox = QtWidget.QCheckBox(col)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.onToggle)
            layout.addWidget(checkbox)
            self.boxes.append(checkbox)
        self.layout = layout
        self.show()


    def showAll(self):
        self.table.showAll()  #Maybe unecessary?
        
        #Update all the checkboxes
        for box in self.boxes:
            box.setChecked(True)

    def hideAll(self):
        self.table.hideAll()  #Maybe unecessary?

        #Update all the checkboxes
        for box in self.boxes:
            box.setChecked(False)

    def onToggle(self, state):
        sender = self.sender()
        self.table.toggleColumn(sender.text(), state>0)

