
#A list of useful constants
#All are in the appropriate SI units

speedOfLight=2.99792458e8
newton=6.67259e-11
planck=6.6260755e-34
hBar=1.05457266e-34 # h/2pi
avogadro=6.0221367e23
boltzmann=1.380658e-23
stefanBoltzmann=5.67e-8 #W.m^-2.K^-4

electronCharge=1.6e-19
electronMass=9.1e-31
protonMass=1.67e-27
neutronMass=1.6749286e-27

planckMass=2.17671e-8
planckLength=1.61605e-35
planckTime=5.39056e-44

#Checked against \S 9.2 of Cambridge Handbook of Physical Formulae
#2010-12-06.
au = 1.495979e11
lightYear = 9.46073e15
parsec = 3.08420e+16
yearInSeconds = 31557600.0

#Values with the comment suffix M15 are taken from
#Mamajek et al (2015, arXiv:1510.07674)
solarMass = 1.988475e30          #M15
solarRadius = 6.957e8            #M15
solarConstant = 1361             #M15, W/m^2
solarLuminosity = 3.828e26       #M15
solarSurfaceTemperature = 5772.  #M15
solarSurfaceLogg = 4.437         #Not checked yet

#Some disagreement about the true value for these quantities.
#See solarconst in utils/mod/ms/ for a good discussion. These
#values from Antia & Basu (2006, ApJ 644 1292)
solarMetalFraction = 0.0172
solarMetalFractionUnc = 0.002

jupiterMass = 1.89818e27             #M15
jupiterRadius = 7.1492e7             #M15  Equatorial radius.

earthMass = 5.97236e24               #M15
earthPolarRadius = 6.3568e6          #M15
earthEquatorialRadius = 6.3781e6     #M15
earthRadius = earthEquatorialRadius  #Arbitary choice. Does it matter?

moonMass=7.3483e22
moonRadius=1.7374e6
moonDistance=3.84400e8

#Equatorial radii, polar is a little less
#Taken from the wikipedia page for each planet.
marsRadius =    3396*1e3
saturnRadius =  60268 * 1e3
neptuneRadius = 24764 * 1e3
uranusRadius =  25559 * 1e3




secondsPerDay=86400.
secondsPerYear=31557600.

#1Jy is 1e-26 W/m^2/Hz
jansky_SI = 1e-26


#Kepler constants
keplerFrameTime_s = 6.53877
keplerShortCadence_s = 58.8489
keplerLongCadence_s = 1765.4679

keplerShortCadence_days = keplerShortCadence_s / float(secondsPerDay)
keplerLongCadence_days = keplerLongCadence_s / float(secondsPerDay)
