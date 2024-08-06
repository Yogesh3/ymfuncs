import healpy as hp
from pixell import enmap
import numpy as np
import os


def readTxt(fname):
    """
    Reads in text file (ideally of filenames) and returns list of rows (ideally a list of the filenames)

    Returns: list
    """    
    names_list = []
    
    with open(fname) as fobj:
        for row in fobj:
            if not row.startswith('#'):
                names_list.append(row.rstrip('\n'))
    
    return names_list
    


def readMapFromFile(mapfile_name):
    """
    Reads in CAR map whose filename is in a text file. Casts to calculation precision.

    Args:
        mapfile_name (string): text file with the full path of the map file
    """
    
    with open(mapfile_name) as name_fobj:
        for line in name_fobj:
            if line[0] == '#':
                continue
            map_name = line.rstrip('\n')

        themap = readMapPrecisely(map_name)

    return themap



def readAlmFromFile(almfile_name):
    """
    Reads in single alm whose filename is in a text file. Casts to calculation precision.

    Args:
        almfile_name (string): text file with the full path of the map file
    """

    with open(almfile_name) as name_fobj:
        for line in name_fobj:
            if line[0] == '#':
                continue
            alm_name = line.rstrip('\n')

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



def getBasename(pathname_with_extension):
    """
    Gets the name of a file without the extension. This is equivalent to pathlib.Path.stem, but that only works for Python 3.4+.

    Parameters
    ----------
    pathname_with_extension : str
        Path to the file. Can be a full path or just the file name with the extension.

    Returns
    -------
    str
        Name of the file without the extension.
    """
    return os.path.splitext(os.path.basename(pathname_with_extension))[0]