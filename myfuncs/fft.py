import numpy as np
from orphics import stats as ostats
from pixell import enmap
from myfuncs import alm as yalm

def xcorr(fft1, fft2=None):
    """
    Cross correlation between two FFTs for flat sky maps.

    Parameters
    ----------
    fft1 : 2d array
        FFT of 2d array
    fft2 : 2d array, optional
        FFT of 2d array, by default None

    Returns
    -------
    2d array
        2d array of the cross correlation
    """
    if fft2 is None:
        fft2 = fft1

    xfft = np.real( fft1 * np.conjugate(fft2) )

    return xfft


def fft2cl(fft, bin_edges, modlmap):
    """
    Bins a 2D FFT to 1D. Designed to convert 2D power spectra to 1D power spectra.

    Parameters
    ----------
    bin_edges : 1d array
        Array of the edges of the ell bins
    fft : 2d array
        2D power spectra
    modlmap : 2d array
        Mapping of each pixel to its wavenumber's magnitude

    Returns
    -------
    1d array, 1d array
        ells_binned (which have length = len(bin_edges) - 1), Cl
    """
    binner = ostats.bin2D(modlmap, bin_edges)
    cents, Cl = binner.bin(fft)

    return cents, Cl


def maps2cl(ell_edges, pixmap1, mask1, pixmap2= None, mask2=None, return_wfac= False):
    if pixmap2 is None:
        pixmap2 = pixmap1

    #Create Composite Mask
    if mask2 is None:
        mask = mask1
    else:
        mask = mask1 * mask2

    #Initializations
    maps = [pixmap1, pixmap2]
    ffts = [] 
    
    #Fourier Transform
    for pixmap in maps:
        ffts.append(enmap.fft(pixmap1, normalize='physical'))

    #Power Spectrum Calculation
    ells_binned, Cls = fft2cl( xcorr(*ffts), ell_edges, pixmap1.modlmap())

    if not return_wfac:
        return ells_binned, Cls
    else:
        return ells_binned, Cls, w2
