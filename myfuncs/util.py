import pdb
import numpy as np
import sys
from astropy import units as u
from astropy.cosmology import Planck15
# from falafel.utils import config
from orphics import cosmology, maps as omaps 
import healpy as hp
import yaml
import myfuncs as ym
from classy_sz import Class

def round_percent(number, nth_non_nine= 1):
    """
    Rounds at the decimal point corresponding to the nth non-nine value. This is useful if you are, e.g. rounding a percent that is very close to the next order of magnitude (99.9994325 => 99.9994). Also, ignores zeros (good for really low percentages).

    Parameters
    ----------
    number : float
        The number to round
    nth_non_nine : int, optional
        The minimum number of non-nine values to have before rounding

    Returns
    -------
    float
        The rounded number
    """    
    #Extract Decimal Points
    str_num = str(number)
    fractional_part = str_num.split('.')[1]

    #Find Decimal Place to Round To
    nth_found_nine = 0
    for decimal_place, digit in enumerate( fractional_part ):
        if int( digit ) != 9 and int( digit ) != 0:
            nth_found_nine += 1
            if nth_found_nine == nth_non_nine:
                break

    return round(number, decimal_place+1)



def deg2fsky(sq_deg):
    """
    Converts square degrees on the sky to the sky fraction.

    Parameters
    ----------
    sq_deg : float
        Area in square degrees

    Returns
    -------
    float
        sky fraction
    """    
    return sq_deg * (np.pi/180)**2 / (4 * np.pi)    



def intersection(main_array, other_array, return_indices= False):
    """
    Mutual elements between two array-like objects (they're cast to arrays here).

    Parameters
    ----------
    main_array : 1darray
        The returned elements are from this array
    other_array : 1darray
        The other array with elements to match

    Returns
    -------
    1darray
        Array of mutual elements.
    """
    #Get Intersection Masks
    set_other = set(other_array)
    set_main = set(main_array)
    mask_main = np.array([x in set_other for x in main_array])
    mask_other = np.array([x in set_main for x in other_array])

    #Get Indices
    main_idxs = np.arange(len(main_array))[mask_main]
    other_idxs = np.arange(len(other_array))[mask_other]

    if return_indices:
        return main_array[mask_main], (main_idxs, other_idxs)
    else:
        return main_array[mask_main]



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
    #Initializations
    array = np.array(array)
    reference = np.array(reference)
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



def find_duplicates_mask(input_array):
    """
    Find all locations where entries in an array are repeated. 
    
    Using Numpy's "unique" function alone doesn't work because it lumps together non-repeated values with the first instance of a repeated value. Using it to obtain the locations of duplications will therefore exclude the first instance of a repeated value.

    Parameters
    ----------
    input_array : array
        The array. Note that this must be an array b/c this function uses numpy's "shape" attribute.

    Returns
    -------
    array
        Mask that selects every instance of a repeated value in the input array
    """
    #Get All But the First Duplications' Locations
    duplications_mask_no_first = np.ones(input_array.shape, dtype=bool)
    duplications_mask_no_first[ np.unique(input_array, return_index=True)[1] ] = False   # selects unique values and FIRST instance of repeated values

    #Get Values That Repeat
    duplicated_values = np.unique(input_array[duplications_mask_no_first])

    #Get All Instances of Repeated Values
    duplications_mask_all = np.zeros(input_array.shape, dtype=bool)
    for duplication in duplicated_values:
        duplicates_idxs = np.argwhere(input_array == duplication)
        duplications_mask_all[duplicates_idxs] = True

    return duplications_mask_all



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



def get_methods(object, spacing=20):
    """
    Print the methods of an object with descriptions. Copied/pasted from https://stackoverflow.com/questions/34439/finding-what-methods-a-python-object-has.

    Parameters
    ----------
    object : object type
        The object you want the methods of
    spacing : int, optional
        _description_. By default 20
    """

    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except Exception:
            methodList.append(str(method_name))
    processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(str(method.ljust(spacing)) + ' ' + processFunc(str(getattr(object, method).__doc__)[0:90]))
        except Exception:
            print(method.ljust(spacing) + ' ' + ' getattr() failed')



def percentDiscrepancy(exp, ref):
    return (exp - ref) / ref * 100



def SN(signal, noise):
    return np.sqrt(np.nansum( signal**2 / noise**2 ))


def cumSN(signal, noise):
    return np.sqrt(np.nancumsum( signal**2 / noise**2 ))



def LenzMasksDict():
    dust_mask = {}

    dust_mask['1.5'] = '1.5e+20_gp20'
    dust_mask['1.8'] = '1.8e+20_gp20'
    dust_mask['2.0'] = '2.0e+20_gp20'
    dust_mask['2.5'] = '2.5e+20_gp20'
    dust_mask['3.0'] = '3.0e+20_gp40'
    dust_mask['4.0'] = '4.0e+20_gp40'

    return dust_mask



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
    p14_dict['omega_ncdm'] = 0.00064
    p14_dict['N_ur'] = 2.0328
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
    p_hm_dict['M_min_HOD_cib'] = pow(10.,10)  # was M_min_HOD_cib

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



def getClassyKappa(params={},
                   pop_params={}, 
                   save_to_yaml= False):
    """
    Return kappa auto spectrum from classy_sz.

    Parameters
    ----------
    params : dict, optional
        Any parameters you want to pass to classy_sz, by default {}
    pop_params : dict, optional
        Names of classy_sz parameters to ignore, by default empty
    save_to_yaml : bool, optional
        Save all of the set parameters to a yaml file on niagara project, by default False

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
    # if 'output' in params.keys():
    #     params.pop('output')      # TODO: make general classy_sz function
        # outspec['output'] = outspec['output'] + ',' + params.pop('output')      

    #Create Class Object
    M = Class()
    
    #Pop Parameterss
    all_params = {**default_params, **outspec}
    for parameter_name in pop_params:
        all_params.pop(parameter_name, None) 

    #Add Parameters
    all_params = {**all_params, **params}
        
    #Save Parameters in YAML File
    if save_to_yaml:
        with open(ym.paths['niagara project'] + 'input_data/kappa_params.yaml', 'w') as yamlfile:
            yaml.dump(all_params, yamlfile)

    #Compute Power Spectra
    M.set(all_params)
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



def TranslateClassyCIBNames(params=[]):

    long_names = ['Redshift_evolution_of_dust_temperature',
                  'Dust_temperature_today_in_Kelvins',
                  'Emissivity_index_of_sed',
                  'Power_law_index_of_SED_at_high_frequency',
                  'Redshift_evolution_of_L_M_normalisation',
                  'Normalisation_of_L_M_relation_in_[JyMPc2/Msun]',
                  'Most_efficient_halo_mass_in_Msun',
                  'Size_of_halo_masses_sourcing_CIB_emission',
                 ]

    short_names = ['alpha',
                   'T_o',
                   'beta',
                   'gamma',
                   'delta',
                   'L_o',
                   'M_eff',
                   'var'
                  ]

    translated_params = []
    for param in params:
        if param in long_names:
            from_list = long_names
            to_list = short_names
        elif param in short_names:
            from_list = short_names
            to_list = long_names
        else:
            raise ValueError(f"{param} is not a vaild parameter name (short or long)")

        idx = from_list.index(param)
        translated_params.append( to_list[idx] )

    return translated_params



def getCIBconstraints(dataset='Planck13', constraint_type='mean'):
    params_cib_dict = {}

    if dataset.lower() == 'planck13':
        if constraint_type == 'mean':
            params_cib_dict['Redshift_evolution_of_dust_temperature'] =  0.36
            params_cib_dict['Dust_temperature_today_in_Kelvins'] = 24.4
            params_cib_dict['Emissivity_index_of_sed'] = 1.75
            params_cib_dict['Power_law_index_of_SED_at_high_frequency'] = 1.7
            params_cib_dict['Redshift_evolution_of_L_M_normalisation'] = 3.6
            params_cib_dict['Most_efficient_halo_mass_in_Msun'] = 10**12.6
            params_cib_dict['Normalisation_of_L_M_relation_in_[JyMPc2/Msun]'] = 6.4e-8
            params_cib_dict['Size_of_halo_masses_sourcing_CIB_emission'] = 0.5

        elif constraint_type == 'err':
            params_cib_dict['Redshift_evolution_of_dust_temperature'] = 0.05
            params_cib_dict['Emissivity_index_of_sed'] = 0.06
            params_cib_dict['Power_law_index_of_SED_at_high_frequency'] = 0.2
            params_cib_dict['Redshift_evolution_of_L_M_normalisation'] = 0.2
            params_cib_dict['Dust_temperature_today_in_Kelvins'] = 1.9
            params_cib_dict['Most_efficient_halo_mass_in_Msun'] = 0.1
            params_cib_dict['Normalisation_of_L_M_relation_in_[JyMPc2/Msun]'] = 1.28e-08

        else:
            raise ValueError("'constraint_type' must be either 'mean' or 'err'")


    elif dataset.lower() == 'viero':      # Viero et al
        if constraint_type == 'mean':
            params_cib_dict['Redshift_evolution_of_dust_temperature'] =  0.2
            params_cib_dict['Dust_temperature_today_in_Kelvins'] = 20.7
            params_cib_dict['Emissivity_index_of_sed'] = 1.6
            params_cib_dict['Power_law_index_of_SED_at_high_frequency'] = 1.7   # not in Viero, so using Planck13
            params_cib_dict['Redshift_evolution_of_L_M_normalisation'] = 2.4
            params_cib_dict['Most_efficient_halo_mass_in_Msun'] = 10**12.3
            params_cib_dict['Normalisation_of_L_M_relation_in_[JyMPc2/Msun]'] = 6.4e-8    # not in Viero, so using Planck13
            params_cib_dict['Size_of_halo_masses_sourcing_CIB_emission'] = 0.3

        elif constraint_type == 'err':
            raise NotImplementedError()

        else:
            raise ValueError("'constraint_type' must be either 'mean' or 'err'")


    else:
        raise NotImplementedError("Need valid data set name")    


    return params_cib_dict



def getClassyCIB(spectra, 
                 nu_list= {353, 545, 857},
                 params={}, 
                 pop_params={}, 
                 emulFlag=False, 
                 save_to_yaml=False
                ):
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
    save_to_yaml : bool, optional
        Save all of the set parameters to a yaml file on niagara project, by default False

    Returns
    -------
    ells, Cls_dict
        Array of ells and a dictionary of Cls. The keys are 'auto' and 'cross', and each of those entries is itself a dictionary indexed by observing frequency as a string (i.e. 'freq' for the cross and 'freqxfreq' for the auto).
    """

    #Get Default Parameters
    default_params = defaultClassyParams()

    #CIB Parameters
    p_CIB_dict = getCIBconstraints()

    #Establish CIB Frequencies
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
    outspec_dict = {'output': ','.join(outspec)}

    #Add Parameters
    all_params = {**default_params, **p_freq_dict, **p_CIB_dict, **outspec_dict}
    for parameter_name in pop_params:
        all_params.pop(parameter_name, None) 
    all_params = {**all_params, **params}
    M.set(all_params)
    # import pdb; pdb.set_trace()

    #Save Parameters in YAML File
    if save_to_yaml:
        with open(ym.paths['niagara project'] + 'input_data/cib_params.yaml', 'w') as yamlfile:
            yaml.dump(all_params, yamlfile)
            
    #Compute Spectra
    if emulFlag:
        M.compute_class_szfast()
    else:
        print("Running class_sz calculations...")
        sys.stdout.flush()
        M.compute()
    
    #Extract Spectra
    Dl_spectra = {}
    if spectra.lower() == 'both' or spectra == 'auto':
        Dl_spectra['auto'] = M.cl_cib_cib()
    if spectra.lower() == 'both' or spectra == 'cross':
        Dl_spectra['cross'] = M.cl_lens_cib()

    #Debugging
    # debug_ell = np.asarray(Dl_spectra['auto']['545x545']['ell'])
    # debug_545 = np.asarray(Dl_spectra['auto']['545x545']['1h']) + np.asarray(Dl_spectra['auto']['545x545']['2h'])
    # debug_data = np.stack([debug_ell, debug_545], axis=1)
    # print("Saving debugging data...")
    # sys.stdout.flush()
    # np.savetxt('/project/r/rbond/ymehta3/debug_545.txt', debug_data)

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



def flux2Tcmb(flux_quantity, freq, type='map'):
    """
    Convert from Janskys/steradians to micro-Kelvins

    Parameters
    ----------
    flux_quantity : array-like
        Thing whose units you want to change from flux.
    freq : float
        Observing frequency
    type : str, optional
        {'map', 'cl'}. By default 'map'

    Returns
    -------
    array-like
        Output object in units of CMB temperature

    Raises
    ------
    ValueError
        Raised if the 'type' parameter wasn't recognized
    """    
    freq = int(freq) * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)

    factor = (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv)

    if type == 'map':
        factor = factor
    elif type == 'cl':
        factor = factor**2
    else:
        raise ValueError('Invalid "type" argument')


    Tcmb_quantity = flux_quantity * factor

    return Tcmb_quantity


def Tcmb2flux(Tcmb_quantity, freq, type='map'):
    """
    Convert from micro-Kelvins to Janskys/steradians

    Parameters
    ----------
    flux_quantity : array-like
        Thing whose units you want to change from flux.
    freq : float
        Observing frequency
    type : str, optional
        {'map', 'cl'}. By default 'map'

    Returns
    -------
    array-like
        Output object in units of CMB temperature

    Raises
    ------
    ValueError
        Raised if the 'type' parameter wasn't recognized
    """    
    freq = int(freq) * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)

    factor = (1. * u.uK).to(u.Jy / u.sr, equivalencies=equiv)

    if type == 'map':
        factor = factor
    elif type == 'cl':
        factor = factor**2
    else:
        raise ValueError('Invalid "type" argument')

    flux_quantity = Tcmb_quantity * factor

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
    """
    Get the names of every unique power spectrum combination possible from given fields.

    Parameters
    ----------
    probes_list : list
        List of names of fields. Their order determines the order of the returned spectra.

    Returns
    -------
    list
        List of names of Cl's. The format is "field1xfield2".
    """
    #Fields -> Cls Matrix 
    Cls_list = []
    for i, probe1 in enumerate(probes_list):
        for j, probe2 in enumerate(probes_list):
            if j >= i:
                Cls_list.append( f'{probe1}x{probe2}' )
                
    return Cls_list


def Cls2Indices(Cls_list):
    """
    Forms a large, multi-spectrum covmat comprised of individual covmats and returns mapping from the names of these individual covmats to their indices within the larger covmat.

    Parameters
    ----------
    Cls_list : list
        List of names of the power spectra you want the covmat for.

    Returns
    -------
    dict
        Mapping from combinations Cl names to indicies in the covmat. The keys are strings of the format "Cl1,Cl2" (names of each individual covmat) and the values are tuples of the indices for the individual covmat.
    """    
    #Cls Matrix -> Covmat
    probe_to_indices = {}
    for i, iCl in enumerate(Cls_list):
        for j, jCl in enumerate(Cls_list):
            probe_to_indices[f'{iCl},{jCl}'] = (i,j)
            
    return probe_to_indices


def Fields2Indices(probes_list):
    """
    Get the indices corresponding to individual covmats of a large, multi-probe covmat from a list of fields. This mapping provides every possible individual covmats that corresponds to every possible combo of every power spectrum possible from the given fields.

    Parameters
    ----------
    probes_list : list
        List of names of all possible fields.

    Returns
    -------
    dict
        Mapping from combinations Cl names to indicies in the covmat. The keys are strings of the format "Cl1,Cl2" (names of each individual covmat), where "Cl1" and "Cl2" are of the format "field1xfield2", and the values are tuples of the indices for the individual covmat.
    """
    #Fields -> Cls Matrix 
    Cls_list = Fields2Cls(probes_list)
         
    #Cls Matrix -> Covmat
    cov_name_to_indices = Cls2Indices(Cls_list)

    return cov_name_to_indices


def getCovmatInfo(labels, info):
    """
    Given either Cl's or fields, return relevant info for the covmat. This allows the flexibility of specifying the relevent Cl's manually or automatically. 

    Parameters
    ----------
    labels : list
        Names that describe a covmat. Either the names of the power spectra for that covmat explicitly (canonically of the format "Cl1xCl2"), in which case, the names should correspond to the order of individual covmats of the covmat of interest (for instance, when reading left to right across the covmat); or the names of the fields (in this case, it's assumed that the covmat being described is the full covmat corresponding to every possible 4pt combination of the fields).
    info : str
        Type of covmat info you want. Options: "Cls" or "indices".

    Returns
    -------
    list or dict
        If you want "Cls", returns list of names of Cl's. If you want "indices", returns the dictionary that maps from Cl combinations to indices (see 'Cls2Indices' for more info).
    """
    if 'x' in labels[0]:
        labels_type = 'Cls'
    else:
        labels_type = 'fields'

    #Return Cl's
    if info.lower() == 'cls':
        if labels_type == 'Cls':
            return labels
        elif labels_type.lower() == 'fields':
            return Fields2Cls(labels)

    #Return Indices
    if info.lower() == 'indices':
        if labels_type == 'Cls':
            return Cls2Indices(labels)
        elif labels_type == 'fields':
            return Fields2Indices(labels)


    
def getIndividualCovmat(Cl1, Cl2, big_covmat, covmat_labels):    
    """
    Get the covmat that corresponds to a single pair of Cl's from a larger covmat.

    Parameters
    ----------
    Cl1 : str
        Name of the first power spectrum that corresponds to the individual covmat.
    Cl2 : str
        Name of the second power spectrum that corresponds to the individual covmat.
    big_covmat : 2darray
        Array of the larger covmat of covmats containing the individual covmat you're after.
    covmat_labels : list
        Names of either fields or Cl's that describe the "big_covmat" (see "labels" argument of "getCovmatInfo" for more info).

    Returns
    -------
    2darray
        Array corresponding to the individual covmat.
    """
    #Get Indices of Individual Covmat
    index_dict = getCovmatInfo(covmat_labels, 'indices')
    i, j = index_dict[f'{Cl1},{Cl2}']
    
    #Calculate Length of Individual Covmat
    Cls = getCovmatInfo(covmat_labels, 'cls')
    N_Cls = len(Cls)
    indiv_covmat_len = int( len(big_covmat) / N_Cls )
    
    #Get Individual Covmat
    indiv_covmat = big_covmat[i*indiv_covmat_len : (i+1)*indiv_covmat_len  ,  j*indiv_covmat_len : (j+1)*indiv_covmat_len]

    return indiv_covmat



def sliceCovmat(start_Cl1, start_Cl2, end_Cl1, end_Cl2, big_covmat, covmat_labels, return_indices= False):    
    """
    Take a 2D slice of a large covmat comprised of individual covmats.

    Parameters
    ----------
    start_Cl1 : str
        Name of the power spectrum corresponding to the starting row of the slicing operation.
    start_Cl2 : str
        Name of the power spectrum corresponding to the starting column of the slicing operation.
    end_Cl1 : str
        Name of the power spectrum corresponding to the ending row of the slicing operation. Inclusive of this end point.
    end_Cl2 : str
        Name of the power spectrum corresponding to the ending column of the slicing operation. Inclusive of this end point.
    big_covmat : 2darray
        Array of the larger covmat of covmats containing the individual covmats you're after.
    covmat_labels : list
        Names of either fields or Cl's that describe the "big_covmat" (see "labels" argument of "getCovmatInfo" for more info).
    return_indices : bool, optional
        List of the indicies involved. Format: [start_row, start_col, end_row, end_col]. By default False.

    Returns
    -------
    2darray (and list, optionally)
        Sliced array (doesn't have to be symmetric). If "return_indices" is True, will also return the indices used for slicing.
    """
    #Get Conversion Between Probe Name and Covmat Index
    probes2indices = getCovmatInfo(covmat_labels, 'indices')
    
    #Calculate Length of Individual Covmat
    Cls = getCovmatInfo(covmat_labels, 'cls')
    N_Cls = len(Cls)
    indiv_covmat_len = int( len(big_covmat) / N_Cls )
    # N_Cls = N_probes * (N_probes - 1) / 2 + N_probes    # upper triangle + diag
    
    #Get Individual Covmat
    istart, jstart = probes2indices[f'{start_Cl1},{start_Cl2}']
    iend, jend = probes2indices[f'{end_Cl1},{end_Cl2}']
    sub_covmat = big_covmat[istart*indiv_covmat_len : (iend+1)*indiv_covmat_len  ,  jstart*indiv_covmat_len : (jend+1)*indiv_covmat_len]

    if return_indices:
        return sub_covmat, [istart,jstart,iend,jend]
    else:
        return sub_covmat



def selectIndivCovmats(select_Cls_list, big_covmat, covmat_labels):
    """
    Create a subcovmat by extracting individual covmats from a big covmat. This can also be used to rearrange the individual covmats of the input big covmat as well (to be clear, this function returns a copy; it doesn't perform the rearrangement in-place).

    Parameters
    ----------
    select_Cls_list : list
        Names of the Cl's that describe the subcovmat you want to create.
    big_covmat : 2darray
        Array of the larger covmat of covmats containing the individual covmats you're after.
    covmat_labels : list
        Names of either fields or Cl's that describe the "big_covmat" (see "labels" argument of "getCovmatInfo" for more info).

    Returns
    -------
    2darray
        The subcovmat.
    """

    sub_covmat = None
    for select_Cl_row in select_Cls_list:
        
        sub_covmat_row = None
        for select_Cl_col in select_Cls_list:

            #Get Individual Covmat
            indiv_covmat = getIndividualCovmat(select_Cl_row, select_Cl_col, big_covmat, covmat_labels)

            #Add Individual Covmat to the Sub Covmat Row
            if sub_covmat_row is None:
                sub_covmat_row = indiv_covmat
            else:
                sub_covmat_row = np.concatenate([sub_covmat_row, indiv_covmat], axis=-1)

        #Add Individual Covmat to the Sub Covmat 
        if sub_covmat is None:
            sub_covmat = sub_covmat_row
        else:
            sub_covmat = np.concatenate([sub_covmat, sub_covmat_row], axis=0)

    return sub_covmat



def cov2corr(covmat):
    """
    Converts individual covariance matrix to a correlation matrix. 

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
