from pprint import pprint
import frmbase.checkimports as ci

class ImportMonitor(dict):
    """Track the edit state of all imports for a module
    
    `check_imports` computes a hash of the contents of every python
    module imported by the module pointed to by `path`. This
    class checks to see if any of those hashes have changed since the 
    last call.

    TODO
    -----
    Should this class take a module name instead of a path?
    """
    def __init__(self):
        self.current_state = dict()

    def have_imports_changed(self, path):
        new_state = ci.check_imports(path)
        new_state = self.update_state_to_hash(new_state)

        flag = False 
        if set(new_state.keys()) != set(self.current_state.keys()):
            flag = True 
        else:
            for k in new_state.keys():
                if new_state[k] != self.current_state[k]:
                    flag = True 
                    break 
        self.current_state = new_state 
        return flag 

    def update_state_to_hash(new_state):
        #Update values from booleans to hashes
        for k in new_state:
            if new_state[k]:
                new_state[k] = ci.get_hash(k)
        return new_state 


