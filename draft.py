import pdb
import numpy as np
import orphics.maps
import time
import healpy as hp
from pixell import enmap, enplot, curvedsky as cs, lensing as plensing
import matplotlib as mpl
import os
from actsims.signal import actsim_root

def import_mpl(machinename, dir=None):
    if machinename.lower() == 'niagara':
        os.environ['MPLCONFIGDIR'] = '/scratch/r/rbond/ymehta3'
    elif machinename.lower() == 'cori':
        os.environ['MPLCONFIGDIR'] = '/global/cscratch1/sd/yogesh3'
    elif machinename.lower() == 'none':
        os.environ['MPLCONFIGDIR'] = dir
    else:
        print("Couldn't import matplotlib. Need valid machine name or writtable path")

    import matplotlib as mpl
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('svg')


def import_plt(machinename, dir=None):
    if machinename.lower() == 'niagara':
        os.environ['MPLCONFIGDIR'] = '/scratch/r/rbond/ymehta3/'
    elif machinename.lower() == 'cori':
        os.environ['MPLCONFIGDIR'] = '/global/cscratch1/sd/yogesh3a'
    elif machinename.lower() == 'none':
        os.environ['MPLCONFIGDIR'] = dir
    else:
        print("Couldn't import pyplot. Need valid machine name or writtable path")

    from matplotlib import pyplot as plt
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('svg')



# def getAlexSims():
    # https://github.com/ACTCollaboration/soapack_configs/blob/master/niagara_act_planck.yml

# def get_kappa_theory(lmax=10e3):
#     """
#     DEPRECATED: I don't know where the theory curve is.
    
#     Grab the kappa autospectrum from Alex's sims from actsims package (https://github.com/ACTCollaboration/actsims/tree/master/data)

#     Parameters
#     ----------
#     lmax : integer, optional
#         maximum ell you want, by default 10e3
#     """
#     #Load Theory Phi Cl
#     # theory_filename = actsim_root + '../data/cosmo2017_10K_acc3_lenspotentialCls.dat'
#     theory_filename = '/home/r/rbond/ymehta3/Software/actsims/data/cosmo2017_10K_acc3_lenspotentialCls.dat'
#     phi_Cl = np.loadtxt(theory_filename, usecols= 5, max_rows= lmax)
#     ells = np.loadtxt(theory_filename, usecols=0, max_rows=lmax)

#     #Convert to Kappa Cl
#     p2k = ells * (ells+1) / 2
#     kappa_Cl = phi_Cl / (p2k**2)
    
#     # return kappa_Cl
#     return phi_Cl


