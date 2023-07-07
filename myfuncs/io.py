import healpy as hp
from pixell import enmap
import numpy as np

def readTxt(fname):
    """
    Reads in text file (ideally of filenames) and returns list of rows (ideally a list of the filenames)

    Returns: list
    """    
    list = []
    
    with open(fname) as fobj:
        for row in fobj:
            list.append(row.rstrip('\n'))
    
    return list
    

def readMapFromFile(mapfile_name):
    """
    Reads in CAR map whose filename is in a text file. Casts to calculation precision.

    Args:
        mapfile_name (string): text file with the full path of the map file
    """
    
    with open(mapfile_name) as name_fobj:
        mapname = name_fobj.readlines()[-1]
        themap = enmap.read_map(mapname)
        themap = np.double(themap)

    return themap


def readAlmFromFile(almfile_name):
    """
    Reads in single alm whose filename is in a text file. Casts to calculation precision.

    Args:
        almfile_name (string): text file with the full path of the map file
    """

    with open(almfile_name) as name_fobj:
        alm_name = name_fobj.readlines()[0]
        alm = hp.read_alm(alm_name)
        alm = np.cdouble(alm)

    return alm


def readMapPrecisely(mapfile_name):
    """
    Reads in CAR map with calculation precision.

    Parameters
    ----------
    mapfile_name : string
        Full path of the map file

    Returns
    -------
    enmap
        The map
    """

    themap = enmap.read_map(mapfile_name)
    themap = np.double(themap)

    return themap


def readAlmPrecisely(almfile_name):
    """
    Reads in single alm with calculation precision.

    Parameters
    ----------
    almfile_name : string
        Full path of the alm file

    Returns
    -------
    1d array
        The alms
    """

    alm = hp.read_alm(almfile_name)
    alm = np.cdouble(alm)

    return alm
