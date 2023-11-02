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
        themap = readMapPrecisely(mapname)

    return themap


def readAlmFromFile(almfile_name):
    """
    Reads in single alm whose filename is in a text file. Casts to calculation precision.

    Args:
        almfile_name (string): text file with the full path of the map file
    """

    with open(almfile_name) as name_fobj:
        alm_name = name_fobj.readlines()[0]
        alm = readAlmPrecisely(alm_name)

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


def getProjectDir(machinename):
    """
    Return the directory with the data for a given machine. This is either a project or scratch directory on a supercomputer, or a directory on a local machine.

    Parameters
    ----------
    machinename : str
        Name of the machine/computer. Options: ['niagara', 'perlmutter', 'legion']

    Returns
    -------
    path : str
        path of the data directory
    """    
    if machinename.lower() == 'niagara':
        path = '/project/r/rbond/ymehta3/'
    elif machinename.lower() == 'perlmutter':
        path = '/pscratch/sd/y/yogesh3/'

    return path