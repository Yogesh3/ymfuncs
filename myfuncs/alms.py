import healpy as hp
import numpy as np


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

