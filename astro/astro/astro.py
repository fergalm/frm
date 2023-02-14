


from tools import toFloat
import numpy as np
import const
import math


def abundanceFromMetalFraction(Z, dZ=None):
    """Compute abundunce, [X/H] from metal fraction, Z/(X+Y+Z)

    Theorists input, and report, the fraction of a solar composition
    that is composed of metals (elements heavier than Helium), as Z,
    with Z between 0 and 1.

    Observers report abundances relative to solar, or [Z/H] =
    log10(Z/H) - log10(Z/H)_solar

    This function converts from one to the other.
    """

    #Can't use toFloat here, because might confuse a bad value with a
    #legitimate one
    try:
        Z = float(Z)
        if dZ is not None:
            dZ = float(dZ)
    except ValueError:
        raise ValueError("Z (%s) and dZ (%s) must be floating point numbers" %
                (Z, dZ))

    Z0 = const.solarMetalFraction
    dZ0 = const.solarMetalFractionUnc

    feh = math.log10(Z) - math.log10(Z0)

    if dZ is None:
        return feh

    dfeh = (dZ/Z)**2 + (dZ0/Z0)**2
    dfeh *= 1/(math.log(10))**2
    dfeh = math.sqrt(dfeh)

    return feh, dfeh



def metalFractionFromAbundance(feh, dFeh=None):
    """Compute metal fraction, Z/(X+Y+Z), from abundunce, [Z/H]

    Theorists input, and report, the fraction of a solar composition
    that is composed of metals (elements heavier than Helium), as Z,
    with Z between 0 and 1.

    Observers report abundances relative to solar, or [Z/H] =
    log10(Z/H) - log10(Z/H)_solar

    This function converts from one to the other.
    """

    #Can't use toFloat here, because might confuse a bad value with a
    #legitimate one
    try:
        feh  = float(feh)
        if dFeh is not None:
            dFeh = float(dFeh)
    except ValueError:
        raise ValueError("feh (%s) and dFeh (%s) must be floating point numbers" %
                (feh, dFeh))

    Z0 = const.solarMetalFraction
    dZ0 = const.solarMetalFractionUnc

    val = 10**(feh)
    Z = Z0*val

    if dFeh is None:
        return Z

    a = val*dZ0
    b = Z0*val*math.log(10) * dFeh
    dZ = math.hypot(a,b)
    return Z, dZ



def estimateEllipsoidalAmplitude_AU(m1_solar, r1_solar, m2_solar, \
 period_days, incl_deg=90):
    """Wrapper for estimateEllipsoidalAmplitude_SI to allow
    user to pass values in astronomical units
    """

    mass1 = m1_solar * const.solarMass
    mass2 = m2_solar * const.solarMass
    radius1 = r1_solar * const.solarRadius

    period = period_days * 86400

    return estimateEllipsoidalAmplitude_SI(mass1, radius1, mass2, \
        period, incl_deg)


def estimateEllipsoidalAmplitude_SI(mass1, radius1, mass2, period, incl_deg=90):
    """Rough estimate of amplitude of ellipsoidal variation

    A companion to a star raises tides on its surface, changing the
    shape of the star to an ellipsoid and causing a
    periodic signal in the lightcurve. This function provides a crude
    estimate of the expected amplitude of this signal for a given system.

    The exact amplitude depends on the limb darkening coeff for a star,
    through the parameter beta. We just fix it to the value for Tres-2.

    Taken from Barclay et al (2012) submitted (on Tres-2).

    For calculation of semi-major axis see Carroll and Ostlie (1st edition)
    Eqn 2.35 (page 52).

    Inputs:
    (all params are floats, and in SI units (kg, m, seconds etc.)
    mass1
    radius1     Parameters of the primary on which tides are raised.
    mass2       Mass of the object raising the tides
    period      Orbital period of system
    incl_deg    Inclination of orbit to line of sight

    Returns:
    Expected fractional amplitude of ellipsoidal variation in
    """

    totalMass = mass1+mass2
    sma = astro.semiMajorAxis_SI(totalMass, period)

    aOverR = sma/radius1

    #For Tres-2, beta is 1.3 (see Barclay et al)
    #Because I'm only doing a quick estimate, hard code this number
    beta = 1.3

    sini = np.sin(np.deg2rad(incl_deg))

    Aell = mass2*beta*sini**2
    Aell /= mass1 * aOverR**3

    return Aell


def orbIncFromImpPar(impactPar, asemi_au, Rstar_solar, \
        impParUnc=None, asemiUnc=None, RstarUnc=None):
    """Compute orbital inclination give impact parameter

    Formula is b = a/R* cos(i)
    inverted gives i = acos(bR/a)

    Return
    if called without uncertainties returns orbital inclination
    in radians

    If called with uncertainties, returns inclination and uncertainty
    in radians.

    """

    #Rename variables and scale to SI
    b = toFloat(impactPar)
    R = toFloat(Rstar_solar)*const.solarRadius
    a = toFloat(asemi_au) * const.au

    val = b*R/a
    inc = np.arccos(val)

    #Now for uncertainties
    if impParUnc is None:
        return inc

    db = impParUnc
    dR = RstarUnc * const.solarRadius
    da = asemiUnc * const.au

    unc = (db/b)**2 + (dR/R)**2 + (da/a)**2
    unc *= (val**2)/(1-val**2)
    unc = np.sqrt(unc)

    return np.array([inc, unc])



def stellarMassFromLoggAndRadius(logg_cgs, radius_solar, loggUnc=None, radiusUnc=None):
    logg_SI = float(logg_cgs) -2   #Convert cgs to SI
    r_SI = float(radius_solar) * const.solarRadius
    mass_SI = (r_SI**2)/const.newton*(10**logg_SI)

    solarMass = mass_SI/const.solarMass
    if loggUnc is None:
        return solarMass

    dr_SI = float(radiusUnc) * const.solarRadius

    a = 2*mass_SI * dr_SI/r_SI
    b = mass_SI * math.log(10)*float(loggUnc)
    dsolarMass = math.hypot(a,b) / const.solarMass

    return solarMass, dsolarMass



def loggCgsFromMassRadius_solar(mass_solar, massUnc, radius_solar, radiusUnc):

    m_SI = mass_solar * const.solarMass
    mUnc = massUnc * const.solarMass

    r_SI = radius_solar * const.solarRadius
    rUnc = radiusUnc * const.solarRadius

    return loggCgsFromMassRadius(m_SI, mUnc, r_SI, rUnc)



def loggCgsFromMassRadius(mass, massUnc, radius, radiusUnc):
    #Gravity in m/s^2
    grav = const.newton*mass/radius**2

    #Convert to log cgs units (+2 converts from SI to cgs)
    logg = np.log10(grav) + 2

    #No need for a constant for uncertainty because we're dealing with logs
    #Note d/dx log10(y) = log10(e)/y
    unc = (massUnc/mass)**2 + (2*radiusUnc/radius)**2
    unc = np.log10(np.exp(1)) * np.sqrt(unc)


    return logg, unc


def radiusSolarFromMassSolarAndLoggCgs(mass_solar, logg_cgs, teffUnc=None, \
    loggUnc=None):

    if teffUnc is not None:
        raise ImplementationError("Uncertainties not available yet")

    mass_SI = mass_solar*const.solarMass
    r = radiusFromMassAndLogg(mass_SI, logg_cgs, teffUnc, loggUnc)

    return r/const.solarRadius


def radiusFromMassAndLogg(mass_SI, logg_cgs, teffUnc=None, \
    loggUnc=None):

    #100 cm/s^2 = 1 m/s^2 => logg(SI) = logg(CGS) -2
    g_SI = 10**(logg_cgs - 2)

    gm = const.newton*mass_SI
    radius = np.sqrt(gm/g_SI)

    if teffUnc is not None:
        raise ImplementationError("Uncertainties not available yet")

    return radius


def radiusFromTeffAndLuminosity_solar(teff_Kelvin, luminosity_solar,
    teffUnc = None, luminosityUnc = None):

    lum_SI = luminosity_solar * const.solarLuminosity
    if teffUnc is None:
        rad = radiusFromTeffAndLuminosity_SI(teff_Kelvin, lum_SI)
        return rad/const.solarRadius
    else:
        dLum_SI = luminosityUnc * const.solarLuminosity
        rad, drad = radiusFromTeffAndLuminosity_SI(teff_Kelvin, lum_SI)
        rad /= const.solarRadius
        drad /= const.solarRadius
        return rad, drad




def radiusFromTeffAndLuminosity_SI(teff_Kelvin, luminosity_SI, teffUnc=None, luminosityUnc=None):

    #Easier to use names
    teff = float(teff_Kelvin)
    lum = float(luminosity_SI)
    sigma = const.stefanBoltzmann

    radius = math.sqrt(lum/(4*math.pi*sigma*(teff**4)))

    if teffUnc is None:
        return radius

    #This code isn't tested.
    #varA = radius/(2*lum) * luminosityUnc
    #varB = -2*radius/teff * teffUnc
    #drad = math.hypot(varA, varB)

    #return radius, drad



def eqTeff(rStar_solar, teffStar_K, asemi_AU, \
    rStarUnc=None, teffUnc=None, asemiUnc=None, redistrib=1, albedo=0.3):
    """
    Compute the equilibrium temperature of a planet

    Default values taken from Batalha (2013)

    """

    radStar = float(rStar_solar) * const.solarRadius
    teffStar = float(teffStar_K)
    asemi = float(asemi_AU) * const.au

    C = redistrib*(1-albedo)
    C = C**.25
    Teq =  teffStar * (radStar/2./asemi)**.5 * C

    if rStarUnc is None:
        return Teq

    drad = float(rStarUnc) * const.solarRadius
    dteff = float(teffUnc)
    dasemi = float(asemiUnc) * const.au

    dTeq = (dteff/teffStar)**2 + (.5*drad/radStar)**2 + (-.5*dasemi/asemi)**2
    dTeq = np.sqrt(dTeq)
    dTeq *= Teq

    return Teq, dTeq



def radialVelocity_SI(m1_SI, m2_SI, period_sec, inclination_deg=90):
    """Max Radial velocity of object of mass m1 if it is orbitted by
    and object of mass m2 with the given period.

    Test:
    .2Mo around 1Mo with period of 20 days give 13892m/s
    Jupiter in 11.2 year orbit is about 10m/s
    """

    sini = np.sin(inclination_deg * np.pi/180)

    val = m2_SI**3
    val /= (m1_SI + m2_SI)**2

    val *= 2*np.pi*const.newton/period_sec

    exponent = 1/3.
    val = val**(exponent) * sini
    return val


def semiMajorAxis_AU(stellarMass_solar, period_days, sMassUnc=None, periodUnc=None):

    sMass = stellarMass_solar*const.solarMass
    period = period_days * 86400

    if sMassUnc is not None:
        sMassUnc *= const.solarMass

    if periodUnc is not None:
        periodUnc *= 86400

    val = semiMajorAxis_SI(stellarMass_solar, period_days, sMassUnc, periodUnc)

    try:
        for i in range(len(val)):
            val[i] = val[i]/const.au
    except TypeError:
        val /= const.au

    return val




def semiMajorAxis(stellarMass_solar, period_days, sMassUnc=None, periodUnc=None):
    """Deprecated. Used semiMajorAxis_AU or _SI instead"""
    #Convert to SI
    mass = float(stellarMass_solar)* const.solarMass
    period = float(period_days) * const.secondsPerDay

    if sMassUnc is not None:
        sMassInc = float(sMassUnc*const.solarMass)

    if periodUnc is not None:
        periodUnc = float(periodUnc) * const.secondsPerDay

    return semiMajorAxis_AU(mass, period, sMassUnc, periodUnc)



def semiMajorAxis_SI(stellarMass, period, sMassUnc=None, periodUnc=None):
    """Compute semi major axis

        Inputs:
        stellarMass     (float) Mass of central body in system (kg)
        period          (float) orbital period (seconds)
        sMassUnc        (float) Uncertainty in stellar mass (kg)
        periodUnc       (float) Uncertainty in period (kg)

        Returns
        Semi major axis in metres. If uncertainties are provided, an
        array of semi-major axis and it's uncertainty are returned.

        Note:
        Adapted from Eqn 2.35 of Carroll & Ostlie 1st Edition (page 52).

        If the mass of the 2nd body is a significant fraction of the
        primary, replace stellar mass with the sum of the masses
        in the input arguments for a more physical answer.
    """

    #Use easy to read names
    G = const.newton
    pi = math.pi
    mass = float(stellarMass)
    period = float(period)

    #Do the calculation
    val = G*mass*period**2
    val /= 4*pi**2
    asemi = val**(1/3.)

    if sMassUnc is None:
        return asemi

    dmass = float(sMassUnc) * const.solarMass
    dperiod = float(periodUnc) * const.secondsPerDay

    val = (dmass/3./mass)**2
    val += (2*dperiod/3./period)**2
    val = asemi*np.sqrt(val)

    return [asemi, val]



def stellarLumFromTeffAndRadius(teff_K, Rstar_solar, teffUnc=None, RstarUnc=None):
    """Compute stellar luminosity

    Formula is L = 4 pi sigma R**2 T**4

    Return
    if called without uncertainties returns luminosity
    in solar units

    If called with uncertainties, returns L and uncertainty
    in solar units

    """

    #Rename variables for convenience
    T = float(teff_K)
    R = float(Rstar_solar) * const.solarRadius
    sigma = const.stefanBoltzmann
    pi = np.pi

    lum = 4*pi*sigma*R**2*T**4
    lum /= const.solarLuminosity

    if teffUnc is None:
        return lum

    dT = float(teffUnc)
    dR = float(RstarUnc) * const.solarRadius

    val = (2*dR/R)**2 + (4*dT/T)**2
    val = np.sqrt(val)
    unc = val * lum

    return lum, val



def reducedMass(mass1, mass2):
    """Calculate the reduced mass of a 2 body system

    Inputs:
    mass(i)     Masses of the two objects.

    Return:
    Reduced mass in same units as mass(i)

    Notes:
    Units not important as long as m1 and m2 are in the same units

    The result in symettric in the arguments, i.e foo(m1, m2) == foo(m2, m1)
    """

    mu = (mass1*mass2)/(mass1+mass2)
    return mu



def wdMassLoggToRadius_m(mass_solar, logg_cgs):
    """Convert logg to a radius estimate"""
    mass_kg = mass_solar * const.solarMass
    logg_SI = logg_cgs - 2

    grav = 10**(logg_SI)
    radius_m = np.sqrt(const.newton * mass_kg/ grav)
    return radius_m

