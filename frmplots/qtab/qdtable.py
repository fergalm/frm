try:
    import PyQt6.QtWidgets as QtWidget
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
except ImportError:
    import PyQt5.QtWidgets as QtWidget
    import PyQt5.QtCore as QtCore
    import PyQt5.QtGui as QtGui

from ipdb import set_trace as idebug 
from colselect import ColumnSelector
from filterbank import FilterBank
from table import TableWidget
import pandas as pd 
    
"""
TODO 
o Filter text box size should match table column size
o Reorg into modules
o Date filter
o Categorical filter
o Better string matching
"""

class QdTable:
    """

    The controller in a model-view-controller pattern
    """
    def __init__(self, df, title=None):
        self.model = Model(df)
        self.colSelect = ColumnSelector(df.columns)
        self.filterBank = FilterBank(df)
        self.view = View(self.filterBank, title=title)

        self.view.button.clicked.connect(self.colSelect.toggleVisible)
        self.filterBank.changed.connect(self.onChange)
        self.colSelect.changed.connect(self.onChange)
        self.view.display(df)

        
    def onChange(self):
        selected = self.colSelect.getSelected()
        df = self.model.get(selected, self.filterBank)
        self.view.display(df)
        


class Model:
    def __init__(self, df):
        self.df = df 
        
    def get(self, cols, filterBank):
        if cols is None:
            cols = self.df.columns 
        
        # print(f"cols are {cols}")
        df = self.df[cols]
        # print(f"df has {len(df)} rows")
        
        if filterBank is None:
            return df.copy()

        for filt in filterBank.get_filters():
            df = filt.applyFilter(df)
        return df.copy()
        

        
class View(QtWidget.QDialog):
    def __init__(self, filterBank, num=1000, title="Dataframe", parent=None):
        super().__init__(parent)
        self.filterBank = filterBank 
        
        # app = QtWidget.QApplication.instance()
        # if app is None:
        #     app = QtWidget.QApplication([])

        self.setWindowTitle(title)
        self.create_layout(title, num)
        self.keyReleaseEvent = self.quit
        self.setFocus(True)
        
    def quit(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.hide()

    def display(self, df):
        self.table.display(df)
        self.filterBank.display(df)
        self.setMaximumWidth(self.table.width() + 20)

    def create_layout(self, title, num):
        self.button = QtWidget.QPushButton("Show/hide Columns")
        self.table = TableWidget(num=num)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.button)
        layout.addLayout(self.filterBank)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.show()
    


