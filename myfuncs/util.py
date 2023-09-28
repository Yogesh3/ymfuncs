import numpy as np
from astropy import units as u
from falafel.utils import config
from orphics import cosmology, maps as omaps 
from classy_sz import Class

def ell2ang(ell, angle_unit=None):
    """Convert from harmonic mode ell to an angle with units.

    Parameters
    ----------
    ell : float
        Harmonic mode ell
    angle_unit : str, optional
        Desired unit for the angle, by default None

    Returns
    -------
    astropy Quantity with astropy Units
        Angle with units

    Raises
    ------
    ValueError
        You need to give an appropriate value for the angle's units

    """

    #Possible Units
    units_dict = {}
    units_dict['deg'] = u.deg
    units_dict['rad'] = u.rad
    units_dict['arcmin'] = u.arcmin
    units_dict['arcsec'] = u.arcsec

    if angle_unit == None:
        raise ValueError(f"Valid angle units are {', '.join(units_dict.keys())}")

    #Calculate Angle
    theta = 2 * np.pi / ell * u.rad

    #Convert Unit
    return theta.to(units_dict[angle_unit], equivalencies=u.dimensionless_angles())


def ang2ell(angle, angle_unit=None):
    """Convert from an angle to harmonic mode ell. This function adds the astropy units so you only need numbers and strings to use it.

    Parameters
    ----------
    angle : float
        Value of angle
    angle_unit : str, optional
        String of the unit for the angle, by default None

    Returns
    -------
    int
        Harmonic mode ell

    Raises
    ------
    ValueError
        You need to give an appropriate value for the angle's units
    """

    #Possible Units
    units_dict = {}
    units_dict['deg'] = u.deg
    units_dict['rad'] = u.rad
    units_dict['arcmin'] = u.arcmin
    units_dict['arcsec'] = u.arcsec

    if angle_unit == None:
        raise ValueError(f"Valid angle units are {', '.join(units_dict.keys())}")

    #Convert to Radians
    angle = angle * units_dict[angle_unit]
    angle = angle.to(u.rad, equivalencies=u.dimensionless_angles())

    #Calculate Angle
    return int(2 * np.pi / angle.value)


def nanToZeros(array):
    return np.where(array == np.nan, 0., array)


def str2bool(string):
    valid_true = ['true', 't', 'yes', 'y']
    valid_false = ['false', 'f', 'no', 'n']
    choice = string.lower()

    if choice in valid_true:
        boolean_val = True
    elif choice in valid_false:
        boolean_val = False
    else:
        raise ValueError(f'Valid boolean values are {valid_true + valid_false}')

    return boolean_val


def sort_str_list(l):
    """
    Sorts list of strings of integers.
    """
    int_list = np.fromiter(l, dtype=int)
    string_list = np.sort(int_list).astype(str)

    return string_list



def percentDiscrepancy(exp, ref):
    return (exp - ref) / ref * 100


def SN(signal, noise):
    return np.sqrt(np.sum( signal**2 / noise**2 ))


def get_theory_dicts(nells=None,lmax=9000,grad=True):

    #Initialize
    if nells is None: nells = {'TT':0,'EE':0,'BB':0}    # noise (dimensionless)
    ls = np.arange(lmax+1)
    ucls = {}
    tcls = {}
    unlensedcls = {}

    #Load Theory
    thloc = config['data_path'] + config['theory_root']
    theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    ells,gt,ge,gb,gte = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1,2,3,4])

    #Repackage Theory into Dictionaries
    ucls['TT'] = omaps.interp(ells,gt)(ls) if grad else theory.lCl('TT',ls)
    ucls['TE'] = omaps.interp(ells,gte)(ls) if grad else theory.lCl('TE',ls)
    ucls['EE'] = omaps.interp(ells,ge)(ls) if grad else theory.lCl('EE',ls)
    ucls['BB'] = omaps.interp(ells,gb)(ls) if grad else theory.lCl('BB',ls)
    unlensedcls['TT'] = theory.uCl('TT',ls)
    unlensedcls['TE'] = theory.uCl('TE',ls)
    unlensedcls['EE'] = theory.uCl('EE',ls)
    unlensedcls['BB'] = theory.uCl('BB',ls)
    ucls['kk'] = theory.gCl('kk',ls)     # this doesn't exist
    tcls['TT'] = theory.lCl('TT',ls) + nells['TT']
    tcls['TE'] = theory.lCl('TE',ls)
    tcls['EE'] = theory.lCl('EE',ls) + nells['EE']
    tcls['BB'] = theory.lCl('BB',ls) + nells['BB']

    return ls, unlensedcls, ucls, tcls



#DEPRECATED!
#Mat's version in orphics is better documented, can take pre-calculated pixel area maps, and works for FFTs and equal area maps
# def wn(mask1, n1, mask2=None, n2=None):
#     """TODO: check pixel area average"""
#     pmap = orphics.maps.psizemap(mask1.shape, mask1.wcs)
#     if mask2 is None:
#         output = np.sum(mask1**n1 * pmap) /np.pi / 4.
#     else:
#         output = np.sum(mask1**n1 * mask2**n2 * pmap) /np.pi / 4.
#     return output

def getClassyCIB(spectra, nu_list, params={}, emulFlag=False, kappaFlag=False):
    """Wrapper for classy_sz calculations of CIB auto and CIB x lensing theory spectra.

    Parameters
    ----------
    spectra : str
        Which CIB spectra do you want? Options: 'auto', 'cross', 'both'
    nu_list : float
        List of observing frequencies as numbers in GHz.
    params : dict, optional
        Dictionary of classy_sz parameters, by default empty
    emulFlag : bool, optional
        Use the cosmopower emulators?, by default False
    kappaFlag : bool, optional
        return kappa autospectrum?, by defualt False

    Returns
    -------
    ells, Cls_dict
        Array of ells and a dictionary of Cls. The keys are 'auto' and 'cross', and each of those entries is itself a dictionary indexed by observing frequency as a string.
    """

    if spectra.lower() not in ['both', 'cross', 'auto']:
        raise ValueError("Your 'spectra' variable is incorrect")

    #Parameters for Cosmology Planck 14, https://arxiv.org/pdf/1303.5076.pdf, best-fit
    p14_dict={}
    p14_dict['h'] = 0.6711 
    p14_dict['omega_b'] = 0.022068
    p14_dict['Omega_cdm'] = 0.3175 - 0.022068/p14_dict['h']/p14_dict['h']
    p14_dict['A_s'] = 2.2e-9
    p14_dict['n_s'] = .9624
    p14_dict['k_pivot'] = 0.05
    p14_dict['tau_reio'] = 0.0925
    p14_dict['N_ncdm'] = 1
    p14_dict['N_ur'] = 0.00641
    p14_dict['deg_ncdm'] = 3
    p14_dict['m_ncdm'] = 0.02
    p14_dict['T_ncdm'] = 0.71611

    p_hm_dict = {}

    p_hm_dict['mass function'] = 'T10'
    p_hm_dict['concentration parameter'] = 'D08'
    p_hm_dict['delta for cib'] = '200m'
    p_hm_dict['hm_consistency'] = 1
    p_hm_dict['damping_1h_term'] = 0
    # Precision
    p_hm_dict['pressure_profile_epsabs'] = 1.e-8
    p_hm_dict['pressure_profile_epsrel'] = 1.e-3
    # HOD parameters for CIB
    p_hm_dict['M_min_HOD'] = pow(10.,10)  # was M_min_HOD_cib

    #Grid Parameters
    # Mass bounds
    p_hm_dict['M_min'] = 1e8 * p14_dict['h']    # was M_min_cib
    p_hm_dict['M_max'] = 1e16 * p14_dict['h']   # was M_max_cib
    # Redshift bounds
    p_hm_dict['z_min'] = 0.07
    p_hm_dict['z_max'] = 6. # fiducial for MM20 : 6
    p_hm_dict['freq_min'] = 10.
    p_hm_dict['freq_max'] = 5e4 # fiducial for MM20 : 6
    p_hm_dict['z_max_pk'] = p_hm_dict['z_max']

    #Precision Parameters
    # Precision for redshift integal
    p_hm_dict['redshift_epsabs'] = 1e-40#1.e-40
    p_hm_dict['redshift_epsrel'] = 1e-4#1.e-10 # fiducial value 1e-8
    # Precision for mass integal
    p_hm_dict['mass_epsabs'] = 1e-40 #1.e-40
    p_hm_dict['mass_epsrel'] = 1e-4#1e-10
    # Precision for Luminosity integral (sub-halo mass function)
    p_hm_dict['L_sat_epsabs'] = 1e-40 #1.e-40
    p_hm_dict['L_sat_epsrel'] = 1e-3#1e-10
    # Multipole array
    p_hm_dict['dlogell'] = 1
    p_hm_dict['ell_max'] = 3968.0
    p_hm_dict['ell_min'] = 2.0

    #CIB Parameters
    p_CIB_dict = {}
    p_CIB_dict['alpha'] =  0.36
    p_CIB_dict['T_o'] = 24.4
    p_CIB_dict['beta'] = 1.75
    p_CIB_dict['gamma'] = 1.7
    p_CIB_dict['delta'] = 3.6
    p_CIB_dict['M_eff'] = 10**12.6
    p_CIB_dict['L_o'] = 6.4e-8
    p_CIB_dict['sigma_sq'] = 0.5

    # nu_list = [353,545,857]
    nu_list_str = str(nu_list)[1:-1]  # Note: this must be a single string, not a list of strings!

    #Frequency Parameters
    p_freq_dict = {}
    p_freq_dict['cib_frequency_list_num'] = len(nu_list)
    p_freq_dict['cib_frequency_list_in_GHz'] = nu_list_str

    #Flux Cuts
    cib_fcut_dict = {}

    #Planck flux cut, Table 1 in https://arxiv.org/pdf/1309.0382.pdf
    cib_fcut_dict['100'] = 400
    cib_fcut_dict['143'] = 350
    cib_fcut_dict['217'] = 225
    cib_fcut_dict['353'] = 315
    cib_fcut_dict['545'] = 350
    cib_fcut_dict['857'] = 710
    cib_fcut_dict['3000'] = 1000

    def _make_flux_cut_list(cib_flux, nu_list):
        """
        Make a string of flux cut values for given frequency list to pass into class_sz
        Beware: if frequency not in the flux_cut dictionary, it assigns 0
        """
        cib_flux_list = []
        keys = list(cib_flux.keys())
        for i,nu in enumerate(nu_list):
            if str(nu) in keys:
                cib_flux_list.append(cib_flux[str(nu)])
            else:
                cib_flux_list.append(0)
        return cib_flux_list

    #Format Flux Cuts
    cib_flux_list = _make_flux_cut_list(cib_fcut_dict, nu_list)

    #Add Flux Cuts
    p_freq_dict['cib_Snu_cutoff_list [mJy]'] = str(list(cib_flux_list))[1:-1]
    p_freq_dict['has_cib_flux_cut'] = 1

    # M.set({# class_sz parameters:
    #     'output':'lens_cib_1h,lens_cib_2h', 
        
    #     #CIB Parameters
    #     'Redshift evolution of dust temperature' :  0.36,
    #     'Dust temperature today in Kelvins' : 24.4,
    #     'Emissivity index of sed' : 1.75,
    #     'Power law index of SED at high frequency' : 1.7,
    #     'Redshift evolution of L - M normalisation' : 3.6,
    #     'Most efficient halo mass in Msun' : 10.**12.6,
    #     'Normalisation of L - M relation in [Jy MPc2/Msun]' : 6.4e-8,
    #     'Size of of halo masses sourcing CIB emission' : 0.5,

    #     #M_min_HOD is the threshold above which nc = 1:
    #     'M_min_HOD' : 10.**10,

    #     'M_min' : 1e10*common_settings['h'],
    #     'M_max' : 1e16*common_settings['h'],
    #     'z_min' : 0.06,
    #     'z_max' : 15,

    #     ### Precision
    #     #redshift_epsabs : 1.0e-40
    #     #redshift_epsrel : 0.0005
    #     #mass_epsabs : 1.0e-40
    #     #mass_epsrel : 0.0005
    #     'dell' : 64,
    #     #multipoles_sz : 'ell_mock'
    #     'ell_max' : 3968.0,
    #     'ell_min' : 2.0,
    #     'ndim_masses' : 100,
    #     'ndim_redshifts' : 100,
        
    #     'cib_frequency_list_num' : Nfreq,
    #     #'cib_frequency_list_in_GHz' : '217,353,545,857,3000',
    #     'cib_frequency_list_in_GHz' : '353, 545, 857'
    #     })

    #Create Class Object
    M = Class()
    
    #Add Spectra
    outspec = []
    if spectra.lower() == 'both' or spectra == 'auto':
        outspec.append('cib_cib_1h,cib_cib_2h')
    if spectra.lower() == 'both' or spectra == 'cross':
        outspec.append('lens_cib_1h,lens_cib_2h')
    if kappaFlag:
        outspec.append('lens_lens_1h,lens_lens_2h')
    M.set({'output': ','.join(outspec)})

    #Add Parameters
    M.set(p14_dict)
    M.set(p_hm_dict)
    M.set(p_CIB_dict)
    M.set(p_freq_dict)
    if params:
        M.set(params)
            
    #Compute Spectra
    if emulFlag:
        M.compute_class_szfast()
    else:
        M.compute()
    
    #Extract Spectra
    Dl_spectra = {}
    if spectra.lower() == 'both' or spectra == 'auto':
        Dl_spectra['auto'] = M.cl_cib_cib()
    if spectra.lower() == 'both' or spectra == 'cross':
        Dl_spectra['cross'] = M.cl_lens_cib()
    M.struct_cleanup()
    M.empty()

    ells = []
    Cls_dict = {}
    #Cycle Through CIB Spectra
    for spec_key, Dl_dict in Dl_spectra.items():
        if spec_key.lower() in ['cross', 'both']:
            freq_list = sort_str_list( list(Dl_dict.keys()) )
        else:
            freq_list = Dl_dict.keys()
        Cls_dict[spec_key] = {}

        #Cycle through Frequencies
        for nu in freq_list:
            #Get ells
            if not len(ells):
                ells = np.array(Dl_dict[nu]['ell'])

            #Get Spectra
            Dl_total = np.array(Dl_dict[nu]['1h']) + np.array(Dl_dict[nu]['2h'])
            Cl_total = dl2cl(Dl_total, ells= ells)

            #Save Spectra
            Cls_dict[spec_key][nu] = Cl_total
        
    #Get Kappa Autospectrum
    if kappaFlag:
        Dl_phi = M.cl_lens_lens()
        Cl_phi = dl2cl(Dl_phi, ells= ells)
        Cls_dict['lens'] = Cl_phi

    return ells, Cls_dict
    
    

def phi2kappa(phi, type= 'spectrum', ells= None):

    if not ells:
        ells = np.arange(len(phi))

    factor = ells * (ells + 1.) / 2.

    if type == 'spectrum':
        kappa = factor**2 * phi
    elif type == 'map':
        kappa = factor**2 * phi
    else:
        raise ValueError('Invalid "type" argument')

    return kappa


def dl2cl(Dl, ells= None):
    
    if ells is None:
        ells = np.arange(len(Dl))
    
    factor = ells * (ells+1) / (2*np.pi)
    
    return Dl / factor



def bin_cen2edg(centers, dbins= None):
    """
    Shifts from midpoints of bins to the edges of the bins (inclusive of the lower and upper endpoints).

    Parameters
    ----------
    centers : 1darray
        Midpoints of bins
    dbins : 1darray, optional
        Size of each bin if you have unevenly spaced bins. By default None

    Returns
    -------
    1darray
        Edges of the bins (including both the lowest and highest edges). Length is 1+len(centers)
    """
    if dbins is None:
        delta = centers[1] - centers[0]
        dbins = np.ones(centers.shape) * delta

    right_edges = centers - dbins/2
    edges = np.append(right_edges, right_edges[-1] + delta)  # doing this instead of centers[-1] * delta/2 avoids issues with odd deltas

    return edges


def bin_edg2cen(edges):
    """
    Shifts from the edges of bins to their midpoints. Works for unevenly sized bins.

    Parameters
    ----------
    edges : 1darray
        Edges of the bins, including the left edge of the first bin and right edge of the last bin.

    Returns
    -------
    1darray
        Midponts of the bins. Length is len(edges) - 1.
    """
    return (edges[1:] + edges[:-1]) / 2


def binning(binsize, xdata, ydata, start='midpoint'):
    """
    Bins x and y data. Handles NaNs just fine. Only works for bins of equal length. For arbitrary or uneven spacing (e.g. log), use orphics.
    """

    #Bin xdata
    if start.lower() == 'midpoint':
        midpoint = binsize//2 - 1              # midpoint if binsize = odd and to the left of midpoint if binsize = even
        xbins = xdata[midpoint : (xdata.size//binsize) * binsize : binsize]
    elif start.lower() == 'left':
        xbins = xdata[: (xdata.size//binsize) * binsize : binsize] 
    else:
        raise ValueError('Need a valid')
    
    #Bin ydata
    ybins = ydata[:(ydata.size//binsize) * binsize]    # drop the last bin if it's too small
    ybins = np.nanmean(ybins.reshape(-1, binsize), axis=-1)

    return xbins, ybins
