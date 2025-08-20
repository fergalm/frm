try:
    import PyQt6.QtWidgets as QtWidget
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
except ImportError:
    import PyQt5.QtWidgets as QtWidget
    import PyQt5.QtCore as QtCore
    import PyQt5.QtGui as QtGui

from colfilters import NumericFilter, StringFilter


class FilterBank(QtWidget.QHBoxLayout):
    changed = QtCore.Signal()
    
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.bank = self.createFilterBank(df)
        self.setLayout()
        
    def setLayout(self):
        for filt in self.bank.values():
            self.addWidget(filt)
        self.setSpacing(0)
        
    def display(self, cols):
        for fname in self.bank:
            if fname in cols:
                self.bank[fname].show()
            else:
                self.bank[fname].hide()
                
    def get_filters(self):
        return list(self.bank.values())
    
    def show(self, col):
        self.bank[col].show()

    def hide(self, col):
        self.bank[col].hide()
        
    def createFilterBank(self, df):
        filters = [NumericFilter, StringFilter]
        out = {}
        
        for c in df.columns:
            for f in filters:
                if f.validate(None, df, c): #TODO cleanup
                    out[c] = f(c)
                    break
            out[c].changed.connect(self.changeFunc)
        return out 

    def changeFunc(self):
        self.changed.emit()
        
