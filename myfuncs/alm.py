import healpy as hp
import numpy as np
import pymaster as nmt
from myfuncs import utils as yutils


def idx2lm(aidx):
    lmax = hp.Alm.getlmax(len(aidx))
    alms = np.zeros((lmax+1, lmax+1), dtype=np.complex128)

    ells, ems = hp.Alm.getlm(lmax)
    idxs = hp.Alm.getidx(lmax, ells, ems)
    alms[ells, ems] = aidx[idxs]
    
    return alms


def lm2idx(alm):
    lmax = alm.shape[0] - 1
    
    ells, ems = hp.Alm.getlm(lmax)
    idxs = hp.Alm.getidx(lmax, ells, ems)
    
    aidx = np.zeros(len(idxs), dtype=np.complex128)
    aidx[idxs] = alm[ells, ems]
    
    return aidx


def lm2cl(alm, weights=None, normalize=True):
    """Converts from l,m space to ell space. Weighs down the monopole and ignores NaN's. Takes m-dependent weights.

    Args:
        alm (l,m array): harmonic coefficients on l,m grid
        weights (l,m array): weights for every l,m
        normize (bool): if True, normalization as a function of ell is calculated. Choose False if the weights are already normalized.
    """
    #Create Weights
    if weights is None:
        weights = np.tril(np.ones(alm.shape))
        weights = np.where(weights == 0, np.nan, weights)
    
    #Create Normalization
    if normalize:
        norm = np.nansum(2 * weights[:, 1:], axis=-1)  +  weights[:, 0]
    else:
        norm = np.ones(alm.shape[0])
    
    #Weighted Average
    alm_weighted_Cl = np.nansum(2 * alm[:,1:] * weights[:,1:], axis=-1)
    alm_weighted_Cl += alm[:,0] * weights[:,0]
    alm_weighted_Cl /= norm

    return alm_weighted_Cl



def knox_formula_errors(power1, power2, fsky, ells, delta_ell, cross_power = None):
    """Probably in orphics"""

    #Note that the power1 and power2 should include noise, but the cross should not.
    return np.sqrt(1. / ((2. * ells + 1) * fsky * delta_ell ) * ((cross_power**2 if cross_power is not None else 0.) + power1 * power2) )



def getField(signal_map, mask, lmax=4000, **user_kwargs):

    #Presets
    standard_kwargs = {}
    standard_kwargs['spin'] = 0
    standard_kwargs['n_iter'] = 0
    standard_kwargs['masked_on_input'] = True
    standard_kwargs['lmax_sht'] = lmax
    
    #Merge keyword arguments
    kwargs = {**standard_kwargs, **user_kwargs}

    field = nmt.NmtField(mask, [signal_map], wcs= mask.wcs, **kwargs)


def calcSpectrum(map_set1, map_set2= None, decouple= True, wsp_name= None):
    """
    Calculates auto/cross spectrum from CAR maps/mask1s using pymaster.

    If you want a mode-decoupled spectrum, provide the name of the workspace with the appropriate mode-coupling matrix. If you want the coupled spectrum, it can do that, but it doesn't apply any wfactors for you (see myfuncs.utils.wfactor).

    Parameters
    ----------
    map_set1 : list
        [map1, mask1]
    map_set2 : list, optional
        [map2, mask2] (only applicable for cross spectrum), by default None
    decouple : bool, optional
        Calculate the mode-decoupled spectrum, by default True
    wsp_name : str, optional
        Path name of the pymaster workspace object with the mode-coupling matrix, by default None

    Returns
    -------
    1D array
        auto/cross spectrum
    """
    #Get Fields
    f1 = getField(map_set1[0], map_set1[1])
    if map_set2:
        f2 = getField(map_set2[0], map_set2[1])
    else:
        f2 = getField(map_set1[0], map_set1[1])

    #Load MCM
    wsp = nmt.NmtWorkspace()
    if wsp_name:
        wsp.read_from(wsp_name)

    #Coupled Spectrum
    Cl_coup = nmt.compute_coupled_cell(f1, f2)

    #Decoupled Spectrum
    if decouple:
        assert wsp_name, "You want a decoupled spectrum but never gave a workspace with the mode-coupling matrix!"

        Cl_decoup = wsp.decouple_cell(Cl_coup)
        Cl_decoup = Cl_decoup[0,:]

        return Cl_decoup

    else:
        return Cl_coup
