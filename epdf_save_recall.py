import numpy as np
import pickle

#import modify_epdfs
#rom modify_epdfs import epdf_new


#######################################
# epdf save/recall code
#######################################

## Import my modified energy pdfs, define function to find and 
## load appropriate pdf, generate and save new one if needed.

# import any previous energy pdf splines I've made.
# If file doesn't exist, create it and populate it with starter pdfs


try:
    with open('epdf-splines-catalog.pkl', 'rb') as f:
        epdf_splines_catalog = pickle.load(f)
except:
    catalog = dict()
    starter_epdf_1 = epdf_new(3.0,0.0,0.0,0,1)
    starter_epdf_2 = epdf_new(3.2,0.0,0.0,0,1)
    catalog['gamma'] = [3.0,3.2]
    catalog['g'] = [0.0,0.0]
    catalog['mphi'] = [0.0,0.0]
    catalog['spline'] = [starter_epdf_1,starter_epdf_2]
    catalog['mcond'] = [0,0] # neutrino masses case 0: heaviest allowed w/ normal ordering
    catalog['gcond'] = [1,1] # flavor-universal neutrino self-coupling
    with open('epdf-splines-catalog.pkl', 'wb') as f:
            pickle.dump(catalog, f)

# Retrieve appropriate energy pdf spline, or if it's not in the
# catalog, evaluate and add it

def epdf_spline(find_gamma,find_g,find_mphi,find_mcond,find_gcond):
    index_val = len(epdf_splines_catalog['gamma'])
    found = False
    for i in range(0,index_val):
        # round to 4th decimal place - avoid unnecessary replication of splines
        # due to rounding error in late decimal places
        c1 = (np.round(epdf_splines_catalog['gamma'][i],4) == np.round(find_gamma,4))
        c2 = (np.round(epdf_splines_catalog['g'][i],4) == np.round(find_g,4))
        c3 = (np.round(epdf_splines_catalog['mphi'][i],4) == np.round(find_mphi,4))
        c4 = epdf_splines_catalog['mcond'][i] == find_mcond
        c5 = epdf_splines_catalog['gcond'][i] == find_gcond
        if c1 and c2 and c3 and c4 and c5:
            found_spline = epdf_splines_catalog['spline'][i]
            found = True
    
    if found == False:
        # generate new spline...
        print(r'Generating new energy pdf spline: gam =',np.round(find_gamma,2),r'logG =',np.round(np.log10(find_g),2),r'logM/MeV =',np.round(np.log10(find_mphi*1000),2))
        this_spline = epdf_new(find_gamma,find_g,find_mphi,find_mcond,find_gcond)
        # ... and add to catalog...
        epdf_splines_catalog['spline'].append(this_spline)
        epdf_splines_catalog['gamma'].append(find_gamma)
        epdf_splines_catalog['g'].append(find_g)
        epdf_splines_catalog['mphi'].append(find_mphi)
        epdf_splines_catalog['mcond'].append(find_mcond)
        epdf_splines_catalog['gcond'].append(find_gcond)
        found_spline = this_spline
        # ... and save to file
        with open('epdf-splines-catalog.pkl', 'wb') as f:
            pickle.dump(epdf_splines_catalog, f)
    return found_spline


# Wrapper for energy pdf splines: calling this function evaluates
# the signal energy pdf value for each point in sample

def eval_energy_pdf(sample,gamma,g,mphi,mcond,gcond):
    epdf = epdf_spline(gamma,g,mphi,mcond,gcond)
    epdf_on_sample = epdf(sample['log10_energy'])
    return epdf_on_sample

