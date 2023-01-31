"""
Stub of a module to interface with my model spectra libraries 
"""

class AbstractSpectrumCollection():
    def __init__(self, path=None, relpath=None):
        if not path:
            path = get_module_path()
            path = os.path.join(path , relpath)
        self.path = path 
        
        self.flist = glob(os.path.join(self.path, "*"))
        assert self.flist
                    
    def getModelList(self) -> pd.DataFrame:
        """Returns a Dataframe of filenames, along with model parameters"""
        pass 
    
    
    def getModel(self) -> np.ndarray:
        pass 
    
    def interp(self, *pars):
        pass 
    
    
def Bsl02(AbstractSpectrumCollection):
    """
    Exoplanet atmosphere models from 
    Burrows, Sudarsky & Luninie (2002)
    """
    
    def __init__(self, path=None):
        AbstractSpectrumCollection.__init__(path, "spec_mods/bsl/*")
        self.df = self.getModelList()

    def getModelList(self) -> pd.DataFrame:
        """Returns a Dataframe of filenames, along with model parameters"""
        
        if self.df is None:
            pars = lmap(self.getParsFromFile, self.flist)
            df = pd.DataFrame()
            df['mass'] = lmap(lambda x: x[0], pars)
            df['age'] = lmap(lambda x: x[1], pars)
            df['cloudy'] = lmap(lambda x: x[2], pars)
            df['path'] = self.flist 
            self.df = df 
            
        return self.df
    
    def getModel(self, mass, age, cloudy):
        """
        TODO What is output format?
        
        Numpy array or pandas.
        microns or metres
        Jy or mJy
        """
        
        df = self.df 
        idx = df.mass == mass
        idx &= df.age == age 
        idx &= df.cloudy == cloudy 
        
        df = df[idx]
        assert len(df) == 1
        data = np.loadtxt(df[idx].path, skiprows=2)
        return data[:, [2,5]]
        


    def getMassAgeFromBsl02Filename(self, f):
        tokens = f.split('_')
        mass = tokens[1]
        age = tokens[2][:3]
        cloudy = tokens[2][3:]
        return mass, age, cloudy
    
    
    
def get_module_path(append):
    path = os.path.split(__file__)[0]
    path = os.path.join(path, append)
    return path
