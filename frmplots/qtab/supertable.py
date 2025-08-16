import numpy as np

import PyQt5.QtWidgets as QtWidget

import colfilters
import newtable

class SuperTableWidget(QtWidget.QDialog):
    def __init__(self, df, num=1000, parent=None):
        QtWidget.QMainWindow.__init__(self, parent)
        self.df = df
        self.ncol = len(df.columns)

        self.collection = create_filter_collection(df)
        self.collection.changed.connect(self.updateFilters)

        self.table = newtable.TableWidget(df, num=num)
        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.collection)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.show()

    def updateFilters(self):
        idx = self.collection.getFilteredIn()
        print("In main window: ", np.sum(idx), " of ", len(idx))
        self.table.drawFiltered(idx)

    def toggleColumn(self, sender_label, state):
        cols = self.table.df.columns
        for i in range(self.ncol):
            if cols[i] == sender_label:
                if state:
                    self.table.showColumn(i)
                    self.collection.showColumn(i)
                else:
                    self.table.hideColumn(i)
                    self.collection.hideColumn(i)

    def showAll(self):
        for i in range(self.ncol):
            self.table.showColumn(i)
            self.collection.showColumn(i)

    def hideAll(self):
        for i in range(self.ncol):
            self.table.hideColumn(i)
            self.collection.hideColumn(i)

    def getMaxWidth(self):
        return self.table.max_width


def create_filter_collection(df):
    cols = df.columns

    filter_list = []
    for c in cols:
        filter_list.append(create_column_filter(df, c))
        print(c, filter_list[-1])

    collection = colfilters.FilterCollection(filter_list)
    return collection


def create_column_filter(df, c):
    col = df[c]
    num_values = len(set(col))
    if num_values < 10:
        return colfilters.CategoricalFilter(col)

    # idebug()
    dtype = col.dtype
    if dtype == np.dtype('int') or dtype == np.dtype('float'):
        return colfilters.NumericFilter(col)
    elif isinstance(dtype, object):
        return colfilters.StringFilter(col)
