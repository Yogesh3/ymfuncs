import healpy as hp
from pixell import enmap
import numpy as np

def readTxt(fname):
    """
    Reads in text file (ideally of filenames) and returns list of rows (ideally a list of the filenames)

    Returns: len(list), list
    """    
    list = []
    
    with open(fname) as fobj:
        for row in fobj:
            list.append(row.rstrip('\n'))
    
    return len(list), list
    

def readMap(mapfile_name):
    """
    Reads in CAR map from text file.

    Args:
        mapfile_name (string): text file with the full path of the map file
    """
    with open(mapfile_name) as name_fobj:
        mapname = name_fobj.readlines()[-1]
        themap = enmap.read_map(mapname)
        
    return themap


def readAlm(almfile_name):
    """
    Reads in single alm from text file.

    Args:
        almfile_name (string): text file with the full path of the map file
    """

    with open(almfile_name) as name_fobj:
        alm_name = name_fobj.readlines()[0]
        alm = hp.read_alm(alm_name)
        alm = np.cdouble(alm)

    return alm
