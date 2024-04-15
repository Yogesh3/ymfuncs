import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15
from falafel.utils import config
from orphics import cosmology, maps as omaps 
import healpy as hp
from classy_sz import Class
import yaml
import myfuncs as ym

def closest_match(array, reference, return_indices= False):
    """
    Find the values in a reference array that most closely match an aribitrary array. Works for unequal sizes for each array.

    Parameters
    ----------
    array : 1darray
        Arbitrary array
    reference : 1darray
        Reference array
    return_indices : bool, optional
        Whether or not to return the reference array indices that correspond to the matched elements. By default, False

    Returns
    -------
    1darray
        Array of values from reference array that most closely match the elements in the arbitrary array.
    list, optional 
        List of indices in the reference array that correspond to the matched elements
    """
    matched_array = []
    matched_indices = []

    for othervalue in array:
        matched_index = ( np.abs(reference - othervalue) ).argmin()
        matched_indices.append(matched_index)
        matched_array.append(reference[matched_index])

    matched_array = np.array(matched_array)

    if return_indices:
        return matched_array, matched_indices
    else:
        return matched_array



def bin_cen2edg(centers, delta= None):
    """
    Shifts from midpoints of bins to the edges of the bins (inclusive of the lower and upper endpoints).

    Parameters
    ----------
    centers : 1darray
        Midpoints of bins
    delta : 1darray, optional
        Size of each bin if you have unevenly spaced bins. By default None

    Returns
    -------
    1darray
        Edges of the bins (including both the lowest and highest edges). Length is 1+len(centers)
    """
    if delta is None:
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
    return np.sqrt(np.nansum( signal**2 / noise**2 ))



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



#################################################################################################
# Class_sz
#################################################################################################

def defaultClassyParams():
    """
    Get default parameters used when getting classy_sz power spectra.

    Parameters
    ----------
    cib_freqs : list, optional
        List of CIB frequencies. By default, []

    Returns
    -------
    dictionary
        Dictionary of parameters for classy_sz. Keys are strings
    """

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

    #More Params
    p_hm_dict['ndim_masses'] = 150 # important 128 is default ccl value
    p_hm_dict['ndim_redshifts'] = 150
    p_hm_dict['non_linear'] = 'halofit'
    p_hm_dict['perturb_sampling_stepsize']  = 0.005
    p_hm_dict['k_max_tau0_over_l_max'] = 5.

    return {**p14_dict, **p_hm_dict}



def getClassyKappa(params={}):
    """
    Return kappa auto spectrum from classy_sz.

    Parameters
    ----------
    params : dict, optional
        Any parameters you want to pass to classy_sz, by default {}

    Returns
    -------
    ells, dict
        Ells and dictionary of kappa auto spectrum. Keys are 'hm' for halo model-based nonlinear power spectrum and 'hf' for HaloFit-based nonlinear power spectrum.
    """
    # #Parameters for Cosmology Planck 14, https://arxiv.org/pdf/1303.5076.pdf, best-fit
    # p14_dict={}
    # p14_dict['h'] = 0.6711 
    # p14_dict['omega_b'] = 0.022068
    # p14_dict['Omega_cdm'] = 0.3175 - 0.022068/p14_dict['h']/p14_dict['h']
    # p14_dict['A_s'] = 2.2e-9
    # p14_dict['n_s'] = .9624
    # p14_dict['k_pivot'] = 0.05
    # p14_dict['tau_reio'] = 0.0925
    # p14_dict['N_ncdm'] = 1
    # p14_dict['N_ur'] = 0.00641
    # p14_dict['deg_ncdm'] = 3
    # p14_dict['m_ncdm'] = 0.02
    # p14_dict['T_ncdm'] = 0.71611

    # p_hm_dict = {}

    # #Grid Parameters
    # # Redshift bounds
    # p_hm_dict['z_min'] = 0.07
    # p_hm_dict['z_max'] = 6. # fiducial for MM20 : 6
    # p_hm_dict['freq_min'] = 10.
    # p_hm_dict['freq_max'] = 5e4 # fiducial for MM20 : 6
    # p_hm_dict['z_max_pk'] = p_hm_dict['z_max']

    # #Precision Parameters
    # # Precision for redshift integal
    # p_hm_dict['redshift_epsabs'] = 1e-40#1.e-40
    # p_hm_dict['redshift_epsrel'] = 1e-4#1.e-10 # fiducial value 1e-8
    # # Precision for mass integal
    # p_hm_dict['mass_epsabs'] = 1e-40 #1.e-40
    # p_hm_dict['mass_epsrel'] = 1e-4#1e-10
    # # Multipole array
    # p_hm_dict['dlogell'] = 1
    # p_hm_dict['ell_max'] = 3968.0
    # p_hm_dict['ell_min'] = 2.0

    #Get Parameters
    default_params = defaultClassyParams()

    #Spectra 
    outspec = {}
    outspec['output'] = 'lens_lens_1h,lens_lens_2h,lens_lens_hf'
    if 'output' in params.keys():
        params.pop('output')      # TODO: make general classy_sz function
        # outspec['output'] = outspec['output'] + ',' + params.pop('output')      

    #Create Class Object
    M = Class()

    #Add Parameters
    M.set(default_params)
    M.set(outspec)
    if params:
        M.set(params)
        
    #Compute Power Spectra
    M.compute()
    Dl_kappa_dict = M.cl_kk()
    ells = np.array(Dl_kappa_dict['ell'])

    Cls_kappa = {}

    #Get Halo Model Kappa
    Dl_kappa_hm = np.array(Dl_kappa_dict['1h']) + np.array(Dl_kappa_dict['2h'])
    Cl_kappa_hm = dl2cl(Dl_kappa_hm, ells= ells)
    Cls_kappa['hm'] = Cl_kappa_hm

    #Get Halofit Kappa
    Cl_hf = dl2cl(Dl_kappa_dict['hf'], ells= ells)
    Cls_kappa['hf'] = Cl_hf

    M.struct_cleanup()
    M.empty()

    return ells, Cls_kappa


    
def getClassyCIB(spectra, 
                 extra_nu_list,
                 params={}, 
                 pop_params={}, 
                 emulFlag=False, 
                 save_to_yaml=False):
    """Wrapper for classy_sz calculations of CIB auto and CIB x lensing theory spectra.

    Parameters
    ----------
    spectra : str
        Can give just the CIB auto or also add in the CIB x lensing. Options: 'auto', 'cross', 'both'
    extra_nu_list : iterable
        List of observing frequencies as numbers in GHz. This is in addition to the default frequencies of 343, 545, and 857 GHz.
    params : list, optional
        Dictionary of classy_sz parameters, by default empty
    pop_params : dict, optional
        Names of classy_sz parameters to ignore, by default empty
    emulFlag : bool, optional
        Use the cosmopower emulators?, by default False

    Returns
    -------
    ells, Cls_dict
        Array of ells and a dictionary of Cls. The keys are 'auto' and 'cross', and each of those entries is itself a dictionary indexed by observing frequency as a string (i.e. 'freq' for the auto and 'freqxfreq' for the cross).
    """

    #Get Default Parameters
    default_params = defaultClassyParams()

    #CIB Parameters
    p_CIB_dict = {}
    # p_CIB_dict['Redshift evolution of dust temperature'] =  0.36
    p_CIB_dict['alpha'] =  0.36
    p_CIB_dict['Dust temperature today in Kelvins'] = 24.4
    p_CIB_dict['Emissivity index of sed'] = 1.75
    p_CIB_dict['Power law index of SED at high frequency'] = 1.7
    p_CIB_dict['Redshift evolution of L - M normalisation'] = 3.6
    p_CIB_dict['Most efficient halo mass in Msun'] = 10**12.6
    p_CIB_dict['Normalisation of L - M relation in [Jy MPc2/Msun]'] = 6.4e-8
    p_CIB_dict['Size of of halo masses sourcing CIB emission'] = 0.5

    #Establish CIB Frequencies
    nu_list = {353, 545, 857}
    for freq in extra_nu_list:
        nu_list.add(freq) 
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


    #Create Class Object
    M = Class()

    #Add Spectra
    outspec = []
    if spectra.lower() not in ['both', 'cross', 'auto']:
        raise ValueError("Your 'spectra' variable is incorrect. See docstring for accepted values.")
    if spectra.lower() == 'both' or spectra == 'auto':
        outspec.append('cib_cib_1h,cib_cib_2h')
    if spectra.lower() == 'both' or spectra == 'cross':
        outspec.append('lens_cib_1h,lens_cib_2h')
    M.set({'output': ','.join(outspec)})

    #Add Parameters
    all_params = {**default_params, **p_freq_dict, **p_CIB_dict}
    for parameter_name in pop_params:
        all_params.pop(parameter_name, None) 
    all_params = {**all_params, **params}
    M.set(all_params)

    #Save Parameters in YAML File
    if save_to_yaml:
        with open(ym.paths['niagara project'] + 'input_data/cib_params.yaml', 'w') as yamlfile:
            yaml.dump(all_params, yamlfile)
            
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

    ells = []
    Cls_dict = {}
    #Cycle Through CIB Spectra
    for spec_key, Dl_dict in Dl_spectra.items():
        if spec_key.lower() == 'cross':
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
        
    M.struct_cleanup()
    M.empty()

    return ells, Cls_dict
    
    

#################################################################################################
# Power Spectra and Maps
#################################################################################################

def phi2kappa(phi, ells= None, type= 'cl'):

    if ells is None:
        ells = np.arange(len(phi))

    factor = ells * (ells + 1.) / 2.

    if type == 'cl':
        kappa = factor**2 * phi
    elif type == 'alm':
        kappa = hp.almxfl(phi, factor)
    else:
        raise ValueError('Invalid "type" argument')

    return kappa



def dl2cl(Dl, ells= None):
    
    if ells is None:
        ells = np.arange(len(Dl))
    
    factor = ells * (ells+1) / (2*np.pi)
    
    return Dl / factor


def cl2dl(Cl, ells= None):
    
    if ells is None:
        ells = np.arange(len(Cl))
    
    factor = ells * (ells+1) / (2*np.pi)
    
    return Cl * factor



def flux2Tcmb(flux_quantity, freq):
    """
    Convert from Janskys/steradians to micro-Kelvins
    """
    freq = int(freq) * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)

    Tcmb_quantity = flux_quantity * (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv)

    return Tcmb_quantity


def Tcmb2flux(Tcmb_quantity, freq):
    """
    Convert from micro-Kelvins to Janskys/steradians
    """
    freq = int(freq) * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
    flux_quantity = Tcmb_quantity * (1. * u.uK).to(u.Jy / u.sr, equivalencies=equiv)

    return flux_quantity



def knox_formula_errors(auto1, fsky, ells, delta_ell, auto2 = None, cross = None):
    """Probably in orphics"""

    prefactor = 1. / ((2. * ells + 1) * fsky * delta_ell )

    if auto2 is None:
        return np.sqrt(prefactor * 2) * auto1

    else:
        #Note that the power1 and power2 should include noise, but the cross should not.
        return np.sqrt(prefactor * ( (cross**2 if cross is not None else 0.)  +  auto1 * auto2) )


    
#################################################################################################
# Covariance Matrix
#################################################################################################

def Fields2Cls(probes_list):
    #Fields -> Cls Matrix 
    Cls_list = []
    for i, probe1 in enumerate(probes_list):
        for j, probe2 in enumerate(probes_list):
            if j >= i:
                Cls_list.append( f'{probe1}x{probe2}' )
                
    return Cls_list


def Cls2Indices(Cls_list):
    #Cls Matrix -> Covmat
    probe_to_indices = {}
    for i, iCl in enumerate(Cls_list):
        for j, jCl in enumerate(Cls_list):
            probe_to_indices[f'{iCl},{jCl}'] = (i,j)
            
    return probe_to_indices


def Fields2Indices(probes_list):
    #Fields -> Cls Matrix 
    Cls_list = Fields2Cls(probes_list)
         
    #Cls Matrix -> Covmat
    cov_name_to_indices = Cls2Indices(Cls_list)

    return cov_name_to_indices

    
def getIndividualCovmat(Cl1, Cl2, big_covmat, probes_list):    
    #Get Conversion Between Probe Name and Covmat Index
    probes2indices = Fields2Indices(probes_list)
    
    #Calculate Length of Individual Covmat
    N_probes = len(probes_list)
    N_Cls = N_probes * (N_probes - 1) / 2 + N_probes    # upper triangle + diag
    indiv_covmat_len = int( len(big_covmat) / N_Cls )
    
    #Get Individual Covmat
    i, j = probes2indices[f'{Cl1},{Cl2}']
    indiv_covmat = big_covmat[i*indiv_covmat_len : (i+1)*indiv_covmat_len  ,  j*indiv_covmat_len : (j+1)*indiv_covmat_len]

    return indiv_covmat



def cov2corr(covmat):
    """
    Converts individual covariance matrix to a correlation matrix. Does not work for a super matrix of covariance matrices.

    Parameters
    ----------
    covmat : 2d array
        Covariance matrix of side length N_ells

    Returns
    -------
    2d array
        Correlation matrix
    """
    variances = np.diag(covmat)
    corr = covmat / np.sqrt(np.outer(variances, variances))
    
    return corr



def superCov2Corr(Covmat, fields_names_list):
    """
    Converts large covariance matrix (comprised of several smaller covmats) into a large correlation matrix (where each small covmat is normalized to that small covmat's diagonal)

    Parameters
    ----------
    Covmat : 2d array
        Square array of side N_ell * N_indivual_covmats
    fields_names_list : list
        List of names of the fields whose cross spectra's covariances are encapsulated in this super Covmat. This assumes that the Covmat has the covariances between all of the cross spectra possible between these fields. If your Covmat contains only a subset of all possible covariances, too bad.

    Returns
    -------
    2d array
        Correlation matrix
    """    
    #Get Covmat Info
    Cls_names = yutil.Fields2Cls(fields_names_list)
    N_Cls = len(Cls_names)
    cov_cls2indices_dict = yutil.Fields2Indices(fields_names_list)

    Corr = np.zeros(Covmat.shape())
    for iCl in Cls_names:
        for jCl in Cls_names:

            #Get Individual Covmat and Its Indices
            indiv_covmat = yutil.getIndividualCovmat(iCl, jCl, Covmat, fields_names_list)
            i, j = cov_cls2indices_dict[f'{iCl},{jCl}']

            #Get Correlation Matrix
            Corr[i,j] = cov2corr(Covmat[i,j])

    return Corr
