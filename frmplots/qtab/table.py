try:
    import PyQt6.QtWidgets as QtWidget
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
except ImportError:
    import PyQt5.QtWidgets as QtWidget
    import PyQt5.QtCore as QtCore
    import PyQt5.QtGui as QtGui

    
class SortableTableItemWidget(QtWidget.QTableWidgetItem):
    """A normal QTableItem widget except it can sort numerically"""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()

class TableWidget(QtWidget.QTableWidget):
    def __init__(self, num=1000, parent=None):
        super().__init__(10, 2, parent=parent)
        self.max_rows = num
        self.max_cols = 100

    def display(self, df):
        self.set_table_elements(df)
        self.setSortingEnabled(True)
        self.draw_row_guides()

        header = self.horizontalHeader()
        header.sectionClicked.connect(self.draw_row_guides)
        self.set_size_policy()

    def set_table_elements(self, df):

        cols = df.columns
        if len(cols) > self.max_cols:
            cols = cols[:self.max_col]
            
        if len(df) > self.max_rows:
            df = df.iloc[:self.max_rows]

        self.setRowCount(len(df))
        self.setColumnCount(len(cols))
            
        QItem = SortableTableItemWidget #Mnumonic
        for i, key in enumerate(cols):
            col = df[key]
            for j, elt in enumerate(col):
                # print(i, j, elt)
                if elt is None:
                    item = QItem(" ")
                    continue
                item = QItem(str(elt))
                item.setFlags( item.flags() & ~QtCore.Qt.EditRole)
                self.setItem(j, i, item)
    
        self.setHorizontalHeaderLabels(df.columns)

    def draw_row_guides(self):
        """This is called by the HeaderView.isClicked signal"""
        guideStep = 5
    
        nrow = self.rowCount()
        ncol = self.columnCount()
        for i in range(nrow):
            clr = QtGui.QColor('#FFFFFF')
            if i % guideStep == guideStep -1:
                clr = QtGui.QColor('#DDDDFF')
    
            for j in range(ncol):
                item = self.item(i, j)
                if item is not None:
                    item.setBackground(clr)
 
    def set_size_policy(self):
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        # self.resizeRowsToContents()

        width_pix = self.horizontalHeader().length() + self.verticalHeader().width() + 20
        self.setMaximumWidth(width_pix)
        self.setMinimumWidth(width_pix)
