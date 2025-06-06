import numpy as np
from orphics import stats as ostats, maps as orphmaps
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
        Array of the left edges of the ell bins
    fft : 2d array
        2D power spectra
    modlmap : 2d array
        Mapping of each pixel to its wavenumber's magnitude

    Returns
    -------
    1d array, 1d array
        Binned ells and Cls (which have length = len(bin_edges) - 1)
    """
    #Bin
    binner = ostats.bin2D(modlmap, bin_edges)
    cents, Cl = binner.bin(fft)

    #Pad The Cls to Match the Ells
    pad_length = len(cents) - len(Cl) 
    Cl = np.concatenate( [Cl, np.zeros(pad_length)] )

    return cents, Cl


def map2cl(ell_edges, 
           pixmap1, 
           mask1= None, 
           pixmap2= None, 
           mask2=None, 
           apodizeFlag= True, 
           return_wfac= False
         ):

    if pixmap2 is None:
        pixmap2 = pixmap1

    #Create Composite Mask
    if mask1 is None and mask2 is None:
        mask = np.ones(pixmap1.shape)
    elif mask1 is None:
        mask = mask2**2
    elif mask2 is None:
        mask = mask1**2
    else:
        mask = mask1 * mask2

    #Initializations
    maps = [pixmap1, pixmap2]
    ffts = [] 
    
    #Apodize
    if apodizeFlag:
        for current_map in maps:
            #Get Apodized Mask
            taper_percent = 4 # %
            apod_window, _ = orphmaps.get_taper(current_map.shape, current_map.wcs, taper_percent= taper_percent)

            #Apply Apodization
            current_map *= apod_window

            #Adjust Composite Mask
            mask *= apod_window
    
    #Fourier Transform
    for pixmap in maps:
        ffts.append(enmap.fft(pixmap, normalize='physical'))

    #Power Spectrum Calculation
    ells_binned, Cls = fft2cl( xcorr(*ffts), ell_edges, pixmap1.modlmap())

    #Wfactor
    mask = enmap.enmap(mask, wcs= pixmap1.wcs)
    wfac = yalm.wfactor(mask, 1, sht= False)
    Cls /= wfac

    if not return_wfac:
        return ells_binned, Cls
    else:
        return ells_binned, Cls, wfac
