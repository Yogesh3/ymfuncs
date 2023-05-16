import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
from pixell import enplot
import numpy as np


def eshow(x,**kwargs): 
    ''' Wrapper to plot enmaps. Sets defaults for the map images and allows to combine maps.'''
        
    #Combine Multiple Maps into a Single Image?
    if 'combine' in kwargs:
        combineFlag = kwargs.pop('combine')
    else:
        combineFlag = False

    #Default Settings
    if 'ticks' not in kwargs:
        kwargs['ticks'] = 10
    if 'downgrade' not in kwargs:
        kwargs['downgrade'] = 10
    if 'colorbar' not in kwargs:
        kwargs['colorbar'] = True

    #Create Plot(s)
    plots = enplot.get_plots(x, **kwargs)
    if combineFlag:
        plots = enplot.merge_plots(sum(plots))

    enplot.show(plots, method = "ipython")


def getDefaultColors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_color_cycle(cmap, N=None, use_index="auto", iterFlag=False, vmin=0, vmax=1):
    """Get iterable color cycler to pass to plt.rcParams["axes.prop_cycle"]. Shamelessly taken from https://stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle

    Parameters
    ----------
    cmap : str
        Valid name of mpl colormap
    N : int, optional
        Number of colors; by default None
    use_index : bool, optional
        Is the colormap indexible?; by default "auto"
    iterFlag: bool, optional
        Return list instead of cycler object; by default False
    vmin, vmax : float, optional
        Min/Max values for cmap; by default [0, 1]

    Returns
    -------
    Cycler
        Desired color cycler
    """
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = mpl.cm.get_cmap(cmap)

    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, mpl.colors.ListedColormap):
            use_index=True

    if use_index:
        ind = np.arange(int(N)) % cmap.N
        cyler = mpl.cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(vmin, vmax, N))
        cycler = mpl.cycler("color",colors)

    if iterFlag:
        return cycler.by_key()['color']
    else:
        return cycler
