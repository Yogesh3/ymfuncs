from orphics.maps import fsky
from pixell import curvedsky as cs, enmap
import healpy as hp
import numpy as np
from astropy import units as u


def skyPercent(themap):
    """
    Calculates the sky fraction of a map as a percent.
    """
    return fsky(themap, threshold=0.0) * 100


def alm2pixmap(alms, footprint_map):
    footprint = enmap.zeros(footprint_map.shape, wcs=footprint_map.wcs)
    pixmap = cs.alm2map(alms, footprint, tweak=True)

    return pixmap


def read_hp2pix(hp_map_fname, footprint_map, lmax, isBoolMask=False):
    """
    DEPRECATED? See pixell?
    Read healpix map from filename as a CAR map.

    Parameters
    ----------
    hp_map_fname : str
        Name of file
    footprint_map : enamp map
        CAR map to project the healpix map onto
    lmax : int
        Healpy's lmax for the spherical harmonic transform
    isBoolMask : bool, optional
        If it's a boolean mask, it doesn't change to a complex double, by default False

    Returns
    -------
    enmap map
        The map in CAR pixelization
    """

    hp_map = hp.read_map(hp_map_fname)
    alms = hp.sphtfunc.map2alm(hp_map, lmax = lmax)
    if not isBoolMask:
        alms = np.cdouble(alms)
    pix_map = alm2pixmap(alms, footprint_map)

    return pix_map


def get_res(wcs, units='deg'):
    """Gets the resolution of a map from WCS information

    Parameters
    ----------
    wcs : WCS object
        WCS of the map
    units : str, optional
        unit of resolution, by default 'deg'

    Returns
    -------
    resolution : float * astropy unit
        Resolution of the map in degrees

    Raises
    ------
    ValueError
        Invalid unit argument
    """        

    #Get Resolution
    resolution = wcs.wcs.cdelt[1] * u.deg

    #Adjust Units as Necessary
    if units != 'deg':
        if units == 'arcmin':
            resolution = resolution.to(u.arcmin)
        else:
            raise ValueError("Valid units are 'deg' and 'arcmin'.")

    return resolution
