from ipdb import set_trace as idebug 
from frmbase.support import lmap 
import frmastro.const as const
from glob import glob 
import pandas as pd
import numpy as np
import os


"""
A class to handle grids of models.

Most of the code for handling these grids of models is the same
(do synth phot, interpolate the models, etc.). The model specific code
is all in figuring out how to read the models from each file,
how to convert the flux from whatever god awful units the original author
chose into mJy, etc. 

This framework abstracts the common code into an abstract class, then
puts the model specific code into daughter classes.

TODO
Better documentation
Add more models 
Split into multiple files
    The abstract class 
    Bsl02
    KoesterDa,
    ...

"""

class AbstractSpectrumCollection:
    """
    TODO: I should cache models in memory for speed once they're loaded
    """

    def __init__(self, wavelength_units, flux_units, path, isRelPath=True, pattern="*"):

        if isRelPath:
            full_path = get_module_path()
            full_path = os.path.join(full_path , path)
            self.path = full_path
        else:
            self.path = path 

        self.pattern = pattern 
        self.flist = glob(os.path.join(self.path, pattern))
        assert self.flist


        self.wavelength_units = wavelength_units
        self.flux_units = flux_units
        self.modelList = self.getModelList()
        self.cache = {}

    def getModelList() -> pd.DataFrame:
        """Construct a dataframe of model parameters and urls
        
        Implemented in the concrete implementation classes
        """
        raise NotImplementedError("getModelList() is only implemented in daughter classes")

    def loadModelFromUrl(self, url) -> pd.DataFrame:   
        """Load a model from disk
        
        Specific to each model type, and implemented in the concrete 
        implementation classes
        """
        raise NotImplementedError("Loading the model is only implemented in daughter classes")


    def getModel(self, **kwargs)-> pd.DataFrame:
        """Given a set of parameters, return the correct model 

        The workhorse function of the class 

        For example, if getModel returns columns including "teff", and  "logg",
        you would call this function as obj.getModel(teff=12000, logg=8.0)

        getModel calls loadModel, which either returns the model from its cache,
        or loads it from disk using loadModelFromUrl. This last function
        is model type specific. 

        Returns a dataframe of wavelength and flux 
        """
        
        df = self.modelList
        idx = np.ones(len(df), dtype=bool)
        for k in kwargs:
            idx &= df[k] == kwargs[k]
        
        df2 = df[idx]
        if len(df2) == 0:
            raise ValueError(f"No model found with properties of {kwargs}. Try interpolation?")

        if len(df2) > 1:
            raise ValueError(f"Multiple models found with properties of {kwargs}. Disambiguate")

        url = df2[idx].path.iloc[0]
        return self.loadModel(url) 

    def loadModel(self, url) -> pd.DataFrame:
        """Load a model, either from cache, or from a url"""
        if url not in self.cache:
            df = self.loadModelFromUrl(url)
            self.cache[url] = df 
        return self.cache[url]
        
    def getWavelengthUnits(self):
        return self.wavelength_units

    def getFluxUnits(self):
        return self.flux_units

    def getInterpModel(self, **kwargs):
        raise NotImplementedError("Interpolation not implemented yet")

    def getSynthPhotTable(self, filters:dict):
        out = []
        filter_names = filters.keys()
        for i, row in self.modelList.iterrows():
            model = self.loadModel(row.url)
            row = list(row)
            for filter_name in filter_names:
                tracing = filters[filter_name]
                flux = synthPhot(model, tracing)
                row.append(flux)
            out.append(row)

        cols = self.modelList.columns
        cols.append(filter_names)
        out = pd.DataFrame(out, columns=cols)
        return out




class Bsl02(AbstractSpectrumCollection):
    """
    Exoplanet atmosphere models from 
    Burrows, Sudarsky & Lunine (2002)
    """
    
    def __init__(self, path='spec_mods/bsl', isRelPath=True):
        AbstractSpectrumCollection.__init__(self, "um", "mJy", path, isRelPath, "fort*")
        

    def getModelList(self) -> pd.DataFrame:
        """Returns a Dataframe of filenames, along with model parameters"""
        
        pars = lmap(self._getParsFromFilename, self.flist)
        df = pd.DataFrame()
        # idebug()
        df['mass'] = lmap(lambda x: x[0], pars)
        df['age'] = lmap(lambda x: x[1], pars)
        df['cloudy'] = lmap(lambda x: x[2], pars)
        df['path'] = self.flist 

        df['mass'] = df.mass.astype(int)
        df['age'] = df.age.astype(float)
        df['cloudy'] = df.cloudy == 'c'
        self.df = df 
            
        return self.df

    def loadModelFromUrl(self, url):
        data = np.loadtxt(url, skiprows=2)
        out = pd.DataFrame()
        out[f'Wavelength_{self.wavelength_units}'] = data[:,2]
        out[f'Flux_{self.flux_units}'] = data[:,5]
        return out 

    def getModelAtMassAge(self, mass_mj, logAge_Gyr) -> pd.DataFrame:
        """Syntatic sugar"""
        #There is never a cloudy and a non-cloudy model at the same gridpoint
        try:
            return self.getModel(mass=mass_mj, age=logAge_Gyr, cloudy=True)
        except ValueError:
            return self.getModel(mass=mass_mj, age=logAge_Gyr, cloudy=False)

    def _getParsFromFilename(self, f):
        tokens = f.split('_')
        mass = tokens[-2]
        age = tokens[-1][:3]
        cloudy = tokens[-1][3:]
        # import pdb; pdb.set_trace()
        return mass, age, cloudy
    

class KoesterDa(AbstractSpectrumCollection):
    """
    DA White Dwarf models from Detlev Koester. 
    These are the set I got from Atsuko Nitta, used as part of Kleinman 04

    Wavelengths are given in metres. Depending on context you may prefer
    to convert to microns, nanometres or Angstroms. 

    Koester gives his flux in some obscure unit per square centimetre of 
    photosphere. I convert that to flux in mJy assuming a 1 earth radius 
    star at 10pc. You should scale that to an appropriate radius before use.

    When in doubt, 1.5 Re is a good ballpark estimate for the true radius.
    """
    
    def __init__(self, path='spec_mods/nitta', isRelPath=True):
        AbstractSpectrumCollection.__init__(self, "metres", "mJy", path, isRelPath, "da*")
        
    def getModelForTeffLogg(self, teff, logg):
        return self.getModel(teff=teff, logg=logg)

    def getModelList(self) -> pd.DataFrame:
        """Returns a Dataframe of filenames, along with model parameters"""

        pars = lmap(self._getParsFromFilename, self.flist)
        df = pd.DataFrame()
        # idebug()
        df['teff'] = lmap(lambda x: x[0], pars)
        df['logg'] = lmap(lambda x: x[1], pars)
        df['path'] = self.flist 
        return df

    def loadModelFromUrl(self, url):
        data = np.loadtxt(url, skiprows=30)
        data = self.convertKoesterToMilliJy(data, const.earthRadius, 10)
        
        out = pd.DataFrame()
        out[f'Wavelength_{self.wavelength_units}'] = data[:,0]
        out[f'Flux_{self.flux_units}'] = data[:,1]
        return out 


    def _getParsFromFilename(self, f):
        f = os.path.basename(f)
        tokens = f.split('_')
        teff = int(tokens[0][2:])
        logg = float(tokens[1][:-3]) / 100 
        return teff, logg 
    

    def convertKoesterToMilliJy(self, data, radius_m, dist_parsec):
        """Convert Koester's file so the wavelengths are in metres
        and the fluxes in Janskys
        """

        wavelength_ang = data[:,0]
        flux = data[:,1]

        wavel_m = wavelength_ang * 1e-10

        # Koester's flux  (4x Eddington) = 4*PI*Intensity
        # Intensity is flux per solid angle. Flux is Intensity
        # integrated over half a solid angle = PI*Intensity
        flux *= 4

        # Convert from flux in erg/cm^2/s/cm to W/m^2/m
        flux *= 1e-1

        # Convert parsecs to metres
        dist_m = dist_parsec * const.parsec

        flux *= (radius_m / dist_m) ** 2

        flux = convertFlambdaToFnu(wavel_m, flux)
        flux /= const.jansky_SI
        flux *= 1e3  #Convert to mJy

        data[:,0] = wavel_m
        data[:,1] = flux 
        return data


def convertFlambdaToFnu(wavel_m, flux_lam):
    factor = (wavel_m **2) / const.speedOfLight
    flux_nu = flux_lam * factor
    return flux_nu


def get_module_path():
    path = os.path.split(__file__)[0]
    return path 



def test_bsl_smoke():
    obj = Bsl02()
    df = obj.getModelAtMassAge(2, 9.0)
    assert isinstance(df, pd.DataFrame)
    # return df

def test_koester_smoke():
    obj = KoesterDa()
    df = obj.getModelForTeffLogg(12000, 8.00)
    assert isinstance(df, pd.DataFrame)
    # return df
