"""
Stub of a module to interface with my model spectra libraries 

Notes:
See miri_wd_worlds/fergal/py/modelspec.py for some functions to deal with
other, not-yet-implemented models
"""

from frmbase.support import lmap 
import astropy.units as u 
from glob import glob 
import pandas as pd 
import numpy as np 
from astro import synth 
from astro import const 

import os 

class AbstractSpectrumCollection():
    def __init__(self, path=None, relpath=None, suffix=""):
        if not path:
            path = get_module_path()
            path = os.path.join(path , relpath)
        self.path = path 
        
        self.flist = glob(self.path + "*"+suffix)
        assert self.flist
        self.df = self.getModelList()
                    
    def getModelList(self) -> pd.DataFrame:
        """Returns a Dataframe of filenames, along with model parameters"""
        pass 
    
    
    def getModel(self) -> np.ndarray:
        pass 
    
    def interp(self, *pars):
        pass 
    


class  Atmo2020(AbstractSpectrumCollection):
    def __init__(self, path):
        AbstractSpectrumCollection.__init__(self, path, suffix=".txt")

    def getModelList(self):
        pars = lmap(self.getParsFromFilename, self.flist)
        df = pd.DataFrame()
        df['teff'] = lmap(lambda x: x[0], pars)
        df['logg'] = lmap(lambda x: x[1], pars)
        df['path'] = self.flist 
        return df
            
        return self.df

    def getParsFromFilename(self, fn):
        tokens = fn.split('_')
        teff = int(tokens[1][1:])
        logg = float(tokens[2][2:])
        return teff, logg 

    def getModel(self, teff, logg, radius_jup) ->np.ndarray:
        """
        TODO What is output format?
        
        Numpy array or pandas.
        microns or metres
        Jy or mJy
        """
        
        df = self.df 
        idx = df.teff == teff
        idx &= df.logg == logg 
        df = df[idx]
        assert len(df) == 1
        url = df.path.iloc[0]
        data = np.loadtxt(url, skiprows=2)
        data = self._convertTomJy(data, radius_jup)
        return data


    def _convertTomJy(self, atmo, radius_jup=1) -> np.ndarray:
        w_um = atmo[:,0]
        f_flam = atmo[:,1]

        w = w_um * u.micron
        f_lam = f_flam * u.Watt / u.m**2 / u.micron
        fjy = flam_to_jy(w, f_lam)

        #Flux from entire hemisphere at 10pc
        radius = radius_jup * const.jupiterRadius
        dist = 10 * const.parsec

        fjy *= (radius/dist)**2
        fmjy = fjy * 1e3

        out = np.zeros((len(atmo), 2))
        out[:,0] = w_um
        out[:,1] = fmjy        
        return out 

class  Bsl02(AbstractSpectrumCollection):
    """
    Exoplanet atmosphere models from 
    Burrows, Sudarsky & Luninie (2002)
    """
    
    def __init__(self, path=None):
        AbstractSpectrumCollection.__init__(self, path, "spec_mods/bsl/fort")

    def getModelList(self) -> pd.DataFrame:
        """Returns a Dataframe of filenames, along with model parameters"""
        
        pars = lmap(self.getParsFromFilename, self.flist)
        df = pd.DataFrame()
        df['mass'] = lmap(lambda x: x[0], pars)
        df['age'] = lmap(lambda x: x[1], pars)
        df['cloudy'] = lmap(lambda x: x[2], pars)
        df['path'] = self.flist 

        df['mass'] = df.mass.astype(int)
        df['age'] = df.age.astype(float)
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
        # import ipdb; ipdb.set_trace()
        
        df = df[idx]
        assert len(df) == 1
        url = df[idx].path.iloc[0]
        data = np.loadtxt(url, skiprows=2)
        return data[:, [2,5]]
        


    def getParsFromFilename(self, f):
        tokens = f.split('_')
        mass = tokens[-2]
        age = tokens[-1][:3]
        cloudy = tokens[-1][3:]
        # import pdb; pdb.set_trace()
        return mass, age, cloudy
    
    
    
def get_module_path():
    path = os.path.split(__file__)[0]
    return path 



def flam_to_jy(w_u, f_flam):
    """Convert flux in wavelength units to Janskys

    This function operates on wavelengths and fluxes with
    astropy units attached. See loadAtmo2020 for an example of usage
    """
    eqv = u.spectral_density(w_u)

    #Flux in Jy per unit surface area
    fjy = f_flam.to(u.Jansky, eqv)
    return fjy
    