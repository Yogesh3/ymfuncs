import healpy as hp
import numpy as np
import pymaster as nmt
from orphics import maps as orphmaps
from pixell import enmap


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



def wfactor(mask1, n1, mask2=None, n2=None, pmap= None, sht= True, equal_area= False):
    """
    Copied from orphics, but generalized to take 2 masks instead of only one. Both
    masks must be on the same map footprint (only matters for CAR).

    Approximate correction to an n-point function for the loss of power
    due to the application of a mask.

    For an n-point function using SHTs, this is the ratio of 
    area weighted by the nth power of the mask to the full sky area 4 pi.
    This simplifies to mean(mask**n) for equal area pixelizations like
    healpix. For SHTs on CAR, it is sum(mask**n * pixel_area_map) / 4pi.
    When using FFTs, it is the area weighted by the nth power normalized
    to the area of the map. This also simplifies to mean(mask**n)
    for equal area pixels. For CAR, it is sum(mask**n * pixel_area_map) 
    / sum(pixel_area_map).

    If not, it does an expensive calculation of the map of pixel areas based on mask1.
    If this has been pre-calculated, it can be provided as the pmap argument.
    
    """
    assert mask1.ndim==1 or mask1.ndim==2
    if mask2 is not None:
        assert mask2.ndim==1 or mask2.ndim==2

    #Get Pixel Map
    if pmap is None: 
        if equal_area:
            npix = mask1.size
            pmap = 4*np.pi / npix if sht else enmap.area(mask1.shape,mask1.wcs) / npix
        else:
            pmap = orphmaps.psizemap(mask1.shape,mask1.wcs)

    #Create Composite Mask
    mask_tot = mask1**n1  if mask2 is None  else  mask1**n1 * mask2**n2 
    
    return np.sum(mask_tot * pmap) / np.pi/4.  if sht  else  np.sum(mask_tot * pmap) / np.sum(pmap)



def getField(signal_map, mask, lmax=4000, **user_kwargs):
    """Calculates the pymaster field for a given map and its mask. Has several keyword arguments.

    Parameters
    ----------
    signal_map : 2d array
        Map. If it's not masked, set 'masked_on_input'=False and the given mask will be applied.
    mask : 2d array
        Mask
    lmax : int, optional
        Ell max for power spectrum, by default 4000

    Returns
    -------
    field object
        Field object from pymaster
    """
    #Presets
    standard_kwargs = {}
    standard_kwargs['spin'] = 0
    standard_kwargs['n_iter'] = 0
    standard_kwargs['masked_on_input'] = True
    standard_kwargs['lmax_sht'] = lmax
    
    #Merge keyword arguments
    kwargs = {**standard_kwargs, **user_kwargs}

    field = nmt.NmtField(mask, [signal_map], wcs= mask.wcs, **kwargs)

    return field


def calcSpectrum(map_set1, map_set2= None, decouple= True, wsp_name= None):
    """
    Calculates auto/cross spectrum from CAR maps/masks using pymaster.

    If you want a mode-decoupled spectrum, provide the name of the workspace with the appropriate mode-coupling matrix. If you want the coupled spectrum, it can do that, but it doesn't apply any wfactors for you (see myfuncs.alm.wfactor).

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
