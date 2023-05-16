import numpy as np 

"""
Initial-final mass relations for white dwarf stars according to various authors

Input the final, white dwar, mass, and get an estimate of the progenitor, main-sequence
mass. 

These references are all 15 years old at this point, and could do with some 
updating, but for approximate work they should be just fine.



"""


def estimate_main_sequence_lifetime_Gyr(mass_solar, mass_unc=None):
    """
    Based on Eqn 1.88 of "Stellar Interiors: Physical Principles, Structure, and 
    Evolution"  by Hansen & Kawaler 
    """
    age =  10 * mass_solar**(-2.5)
    
    if mass_unc is None:
        return age

    age_unc = mass_unc * 10 * (2.5) * mass_solar ** (-3.5) 
    return age, age_unc
    

def williams07_ifmr(wd_mass_solar, mass_unc=None):
    """
    Taken from Williams & Bolte 2007AJ....133.1490W 
    """
    
    a, da = .132, .017     #//slope = a+bx
    b, db =.33, .07
    
    return linear_ifmr(wd_mass_solar, mass_unc, a, da, b, db)



def linear_ifmr(Mf_solar, dMf, a0, da0, b0, db0):
    """Compute inital progenitor mass from final white dwarf mass using a linear
    initial-final mass ratio
    
    Inputs
    -----------
    Mf_solar, dMf
        (float) Final, WD, mass in solar masses, with uncertainty
    """
    
    Minit_solar = (Mf_solar - a0) / b0
    if dMf is None:
        return Minit_solar 
    
    
    val1 = dMf**2 + da0**2;
    val1 /= b0**2;

    val2 = (Mf_solar - a0)**2 * db0**2;
    val2 /= b0**4;

    Minit_unc = np.sqrt(val1 + val2);
    return Minit_solar, Minit_unc
        




#//Because so many IFMR's are linear, I've abstracted out a function to
#//do this simplest calculation.
#void linearIFMR(double Mf, double dMf, double a0, double da0, double a1, double da1, double *Mi, double *dMi)
#{
    #double val1, val2;
    
    #(*Mi) = (Mf-a0)/a1;
    
    #//If user doesn't request uncertainty, don't calculate it
    #if(dMi != NULL)
    #{
        #val1 = sqr(dMf) + sqr(da0);
        #val1 /= sqr(a1);

        #val2 = sqr(Mf-a0) * sqr(da1);
        #val2 /= pow(a1,4);

        #(*dMi) = sqrt(val1 + val2);
    #}
#}


#void williams07IFMR(double wdmass, double dwdmass, double *starmass, double *dstarmass)
#{
    #double val, dval;
    #double a=.132, da=.017;     //slope = a+bx
    #double b=.33, db=.07;

    #(*starmass) = (wdmass - b)/a;

    #val = (wdmass-b)/sqr(a)*da;
    #dval = sqr(val);

    #val = db/a;
    #dval += sqr(val);

    #val = dwdmass/a;
    #dval += sqr(val);

    #(*dstarmass) = sqrt(dval);
#}



#//Ferrario05 has two fits to the IMFR, I only use the linear case
#// 2005MNRAS.361.1131F 
#void ferrario05IFMR(double wdmass, double dwdmass, double *starmass, double *dstarmass)
#{
    #double val, dval;
    #double a=.10038, da=.00518;     //slope = a+bx
    #double b=.43443, db=.01467;

    #(*starmass) = (wdmass - b)/a;

    #val = (wdmass-b)/sqr(a)*da;
    #dval = sqr(val);

    #val = db/a;
    #dval += sqr(val);

    #val = dwdmass/a;
    #dval += sqr(val);

    #(*dstarmass) = sqrt(dval);
#}



#void kalirai07IFMR(double wdmass, double dwdmass, double *starmass, double *dstarmass)
#{
    #double a0, da0;
    #double a1, da1;
    
    #//y = a0 + a1*x
    #a0=0.394;
    #da0=0.025;
    
    #a1= 0.109;
    #da1= 0.007;
    
    #linearIFMR(wdmass, dwdmass, a0, da0, a1, da1, starmass, dstarmass);
#}



#void dobbie06IFMR(double wdmass, double dwdmass, double *starmass, double *dstarmass)
#{
    #double a0, da0;
    #double a1, da1;
    
    #//y = a0 + a1*x
    #a0=0.289;
    #da0=0.051;
    
    #a1= 0.133;
    #da1= 0.015;
    
    #linearIFMR(wdmass, dwdmass, a0, da0, a1, da1, starmass, dstarmass);
#}
