import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
from pixell import enplot, colorize
import numpy as np
from myfuncs import util as yutil


ibm_colors = [(100/255, 143/255, 255/255),
              (120/255, 94/255, 240/255),
              (220/255, 38/255, 127/255),
              (254/255, 97/255, 0/255),
              (255/255, 176/255, 0/255)]

planck_colors = np.array([(0,   0, 255),
                          (0,   2, 255),
                          (0,   5, 255),
                          (0,   8, 255), 
                          (0,  10, 255), 
                          (0,  13, 255), 
                          (0,  16, 255), 
                          (0,  18, 255), 
                          (0,  21, 255), 
                          (0,  24, 255), 
                          (0,  26, 255), 
                          (0,  29, 255), 
                          (0,  32, 255), 
                          (0,  34, 255), 
                          (0,  37, 255), 
                          (0,  40, 255), 
                          (0,  42, 255), 
                          (0,  45, 255), 
                          (0,  48, 255), 
                          (0,  50, 255), 
                          (0,  53, 255), 
                          (0,  56, 255), 
                          (0,  58, 255), 
                          (0,  61, 255), 
                          (0,  64, 255), 
                          (0,  66, 255), 
                          (0,  69, 255), 
                          (0,  72, 255), 
                          (0,  74, 255), 
                          (0,  77, 255), 
                          (0,  80, 255), 
                          (0,  82, 255), 
                          (0,  85, 255), 
                          (0,  88, 255), 
                          (0,  90, 255), 
                          (0,  93, 255), 
                          (0,  96, 255), 
                          (0,  98, 255), 
                          (0, 101, 255), 
                          (0, 104, 255), 
                          (0, 106, 255), 
                          (0, 109, 255), 
                          (0, 112, 255), 
                          (0, 114, 255), 
                          (0, 117, 255), 
                          (0, 119, 255), 
                          (0, 122, 255), 
                          (0, 124, 255), 
                          (0, 127, 255), 
                          (0, 129, 255), 
                          (0, 132, 255), 
                          (0, 134, 255), 
                          (0, 137, 255), 
                          (0, 139, 255), 
                          (0, 142, 255), 
                          (0, 144, 255), 
                          (0, 147, 255), 
                          (0, 150, 255), 
                          (0, 152, 255), 
                          (0, 155, 255), 
                          (0, 157, 255), 
                          (0, 160, 255), 
                          (0, 162, 255), 
                          (0, 165, 255), 
                          (0, 167, 255), 
                          (0, 170, 255), 
                          (0, 172, 255), 
                          (0, 175, 255), 
                          (0, 177, 255), 
                          (0, 180, 255), 
                          (0, 182, 255), 
                          (0, 185, 255), 
                          (0, 188, 255), 
                          (0, 190, 255), 
                          (0, 193, 255), 
                          (0, 195, 255), 
                          (0, 198, 255), 
                          (0, 200, 255), 
                          (0, 203, 255), 
                          (0, 205, 255), 
                          (0, 208, 255), 
                          (0, 210, 255), 
                          (0, 213, 255), 
                          (0, 215, 255), 
                          (0, 218, 255), 
                          (0, 221, 255), 
                          (6, 221, 254), 
                          (12, 221, 253),
                          (18, 222, 252),
                          (24, 222, 251),
                          (30, 222, 250),
                          (36, 223, 249),
                          (42, 223, 248),
                          (48, 224, 247),
                          (54, 224, 246),
                          (60, 224, 245),
                          (66, 225, 245),
                          (72, 225, 244),
                          (78, 225, 243),
                          (85, 226, 242),
                          (91, 226, 241),
                          (97, 227, 240),
                          (103, 227, 239),
                          (109, 227, 238),
                          (115, 228, 237),
                          (121, 228, 236),
                          (127, 229, 236),
                          (133, 229, 235),
                          (139, 229, 234),
                          (145, 230, 233),
                          (151, 230, 232),
                          (157, 230, 231),
                          (163, 231, 230),
                          (170, 231, 229),
                          (176, 232, 228),
                          (182, 232, 227),
                          (188, 232, 226),
                          (194, 233, 226),
                          (200, 233, 225),
                          (206, 233, 224),
                          (212, 234, 223),
                          (218, 234, 222),
                          (224, 235, 221),
                          (230, 235, 220),
                          (236, 235, 219),
                          (242, 236, 218),
                          (248, 236, 217),
                          (255, 237, 217),
                          (255, 235, 211),
                          (255, 234, 206),
                          (255, 233, 201),
                          (255, 231, 196),
                          (255, 230, 191),
                          (255, 229, 186),
                          (255, 227, 181),
                          (255, 226, 176),
                          (255, 225, 171),
                          (255, 223, 166),
                          (255, 222, 161),
                          (255, 221, 156),
                          (255, 219, 151),
                          (255, 218, 146),
                          (255, 217, 141),
                          (255, 215, 136),
                          (255, 214, 131),
                          (255, 213, 126),
                          (255, 211, 121),
                          (255, 210, 116),
                          (255, 209, 111),
                          (255, 207, 105),
                          (255, 206, 100),
                          (255, 205,  95),
                          (255, 203,  90),
                          (255, 202,  85),
                          (255, 201,  80),
                          (255, 199,  75),
                          (255, 198,  70),
                          (255, 197,  65),
                          (255, 195,  60),
                          (255, 194,  55),
                          (255, 193,  50),
                          (255, 191,  45),
                          (255, 190,  40),
                          (255, 189,  35),
                          (255, 187,  30),
                          (255, 186,  25),
                          (255, 185,  20),
                          (255, 183,  15),
                          (255, 182,  10),
                          (255, 181,   5),
                          (255, 180,   0),
                          (255, 177,   0),
                          (255, 175,   0),
                          (255, 172,   0),
                          (255, 170,   0),
                          (255, 167,   0),
                          (255, 165,   0),
                          (255, 162,   0),
                          (255, 160,   0),
                          (255, 157,   0),
                          (255, 155,   0),
                          (255, 152,   0),
                          (255, 150,   0),
                          (255, 147,   0),
                          (255, 145,   0),
                          (255, 142,   0),
                          (255, 140,   0),
                          (255, 137,   0),
                          (255, 135,   0),
                          (255, 132,   0),
                          (255, 130,   0),
                          (255, 127,   0),
                          (255, 125,   0),
                          (255, 122,   0),
                          (255, 120,   0),
                          (255, 117,   0),
                          (255, 115,   0),
                          (255, 112,   0),
                          (255, 110,   0),
                          (255, 107,   0),
                          (255, 105,   0),
                          (255, 102,   0),
                          (255, 100,   0),
                          (255,  97,   0),
                          (255,  95,   0),
                          (255,  92,   0),
                          (255,  90,   0),
                          (255,  87,   0),
                          (255,  85,   0),
                          (255,  82,   0),
                          (255,  80,   0),
                          (255,  77,   0),
                          (255,  75,   0),
                          (251,  73,   0),
                          (247,  71,   0),
                          (244,  69,   0),
                          (240,  68,   0),
                          (236,  66,   0),
                          (233,  64,   0),
                          (229,  62,   0),
                          (226,  61,   0),
                          (222,  59,   0),
                          (218,  57,   0),
                          (215,  55,   0),
                          (211,  54,   0),
                          (208,  52,   0),
                          (204,  50,   0),
                          (200,  48,   0),
                          (197,  47,   0),
                          (193,  45,   0),
                          (190,  43,   0),
                          (186,  41,   0),
                          (182,  40,   0),
                          (179,  38,   0),
                          (175,  36,   0),
                          (172,  34,   0),
                          (168,  33,   0),
                          (164,  31,   0),
                          (161,  29,   0),
                          (157,  27,   0),
                          (154,  26,   0),
                          (150,  24,   0),
                          (146,  22,   0),
                          (143,  20,   0),
                          (139,  19,   0),
                          (136,  17,   0),
                          (132,  15,   0),
                          (128,  13,   0),
                          (125,  12,   0),
                          (121,  10,   0),
                          (118,   8,   0),
                          (114,   6,   0),
                          (110,   5,   0),
                          (107,   3,   0),
                          (103,   1,   0),
                          (100,   0,   0) 
                        ])/ 255



def get_predefined_colors():
    color_names = []
    color_names.append('ibm')
    color_names.append('planck')

    return color_names



def eshow(the_map, show_img= True, return_img= False, **user_kwargs): 
    """
    Wrapper to plot enmaps. Includes defaults for the image.

    Parameters
    ----------
    the_map : enmap
        Enmap object to plot
    show_img : bool, optional
        Whether or not to show the image. By default True
    return_img : bool, optional
        Whether or not to return the enplot object. By default False

    Returns
    -------
    enplot object, optional
        Plot object.
    """     
        
    #Default Settings
    default_kwargs = {}
    default_kwargs['ticks'] = 10
    default_kwargs['downgrade'] = 10
    default_kwargs['colorbar'] = True
    default_kwargs['mask'] = 0
    total_kwargs = {**default_kwargs, **user_kwargs}

    #Create Plot
    out_plot = enplot.get_plots(the_map, **total_kwargs)

    #Show Plot
    if show_img:
        enplot.show(out_plot, method = "ipython")

    #Return Image
    if return_img:
        return out_plot



def emerge_plots(mapslist, kwargs_list= [], show_img= True, return_img= False):
    """
    Wrapper to merge enmaps into a single overlay plot. While defaults for each map are provided, they don't really make sense for more than 2 maps (all maps >=2 are indentical contours), so make sure to specify how you want each plot to look.

    Parameters
    ----------
    mapslist : list
        List of enmaps.
    kwargs_list : list, optional
        List of dicts with enplot kwargs that specify how each map looks. By default []
    show_img : bool, optional
        Whether or not to show the image. By default True
    return_img : bool, optional
        Whether or not to return the enplot object. By default False

    Returns
    -------
    enplot object, optional
        Plot object of all maps overlaid on each other.
    """     
    #Later Maps Kwargs
    laterplots_kwargs = {}
    laterplots_kwargs['colorbar'] = False
    laterplots_kwargs['no_image'] = True
    laterplots_kwargs['contours'] = 0.1

    for imap, current_map in enumerate(mapslist):

        #Get First Plot
        if imap == 0:
            #Establish Plot Kwargs
            if len(kwargs_list) == 0:
                kwargs = {}
            else:
                kwargs = kwargs_list[imap]

            all_plots = eshow(current_map, 
                              show_img= False,
                              return_img= True,
                              **kwargs
                             )
        
        #Add Later Plots
        else:
            #Establish Plot Kwargs
            if len(kwargs_list) == 0:
                kwargs = laterplots_kwargs
            else:
                kwargs = {**laterplots_kwargs, **kwargs_list[imap]}

            current_plot = eshow(current_map,
                                 show_img= False,
                                 return_img= True,
                                 **kwargs
                                )
        
            all_plots += current_plot

    #Merge Plots
    out_plot = enplot.merge_plots(all_plots)

    #Show Plot
    if show_img:
        enplot.show(out_plot, method = "ipython")

    #Return Image
    if return_img:
        return out_plot



def getCurrentColors():
    """
    Returns list of colors in current color cycle.
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']



def set_cycler(cycler, ax= None):
    """
    Sets the cycler either globally or for a specified axis.

    Parameters
    ----------
    cycler : cycler obj
        General cycler object
    ax : Axis obj, optional
        Axis to set the cycle. By default None
    """
    if ax is None:
        plt.rcParams['axes.prop_cycle'] = cycler
    else:
        ax.set_prop_cycle(cycler)



def get_color_cycle(cmap='tab10', N=None, use_index="auto", iterFlag=False, vmin=0, vmax=1):
    """Get iterable color cycler to pass to plt.rcParams["axes.prop_cycle"]. Shamelessly adapted from https://stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle

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
        Desired color cycler. Will be a list if 'iterFlag' = True
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
        cycler = mpl.cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(vmin, vmax, N))
        cycler = mpl.cycler("color",colors)

    if iterFlag:
        return cycler.by_key()['color']
    else:
        return cycler



def categorical_cmap(nc, nsc, cmap_name="tab10"):
    """
    For a given colormap, adds subcolors of various saturation levels (technically different hues) for each color. Shamelessly adapted from https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10

    Parameters
    ----------
    nc : int
        Number of colors
    nsc : int
        Number of subcolors
    cmap_name : str, optional
        Valid mpl colormap name, by default "tab10"

    Returns
    -------
    Colormap
        Iterable color map with the subcolors

    Raises
    ------
    ValueError
        Can't have more base colors than the number of possible hues in the colormap.
    """
    cmap = plt.get_cmap(cmap_name)

    if nc > cmap.N:
        raise ValueError("Too many categories for colormap.")
    
    #Determine if Continuous or Discrete Colormap
    if cmap.N > 100:
        continous = True
    elif isinstance(cmap, mpl.colors.LinearSegmentedColormap):
        continous = True
    elif isinstance(cmap, mpl.colors.ListedColormap):
        continous = False

    #Get Category Colors
    if continuous:
        ccolors = cmap(np.linspace(0,1,nc))
    else:
        ccolors = cmap(np.arange(nc, dtype=int))

    #Add Subcategory Colors
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    new_cmap = matplotlib.colors.ListedColormap(cols)

    return new_cmap



def custom_cmap( cmap_name= '', 
                 color_list = []
               ):
    """
    Creates a continuous matplotlib colormap from a sequence of finite colors.

    Parameters
    ----------
    cmap_name : str, optional
        Name of predefined color sequence. By default ''.
    color_list : list, optional
        Color sequence to make a colormap from. By default [].

    Returns
    -------
    Colormap
        Mpl continuous colormap.

    Raises
    ------
    ValueError
        Warns if you request a colormap name that wasn't pre-defined.
    """

    if cmap_name not in get_predefined_colors() and not color_list:
        raise ValueError("Need either a pre-defined custom cmap name or color values")

    #Use Pre-defined Colors
    if cmap_name.lower() == 'planck':
        color_list = planck_colors
    elif cmap_name.lower() == 'ibm':
        color_list = ibm_colors
    
    #Create Colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, color_list)

    #Set Default Things
    cmap.set_bad('grey')    # color of missing pixels
    cmap.set_bad('white')    # color of hp.mollview background

    return cmap



def plot_color_gradients(cmap_names_list, title='Colormaps'):
    """
    Plot 1D strips of colormaps. Adapted from https://matplotlib.org/stable/gallery/color/colormap_reference.html

    Parameters
    ----------
    cmap_names_list : string or list of strings
        List of names corresponding to either official MPL colormaps or colormaps defined in this library. Can also be a single string (not wrapped in a list) if just looking at 1 colormap.
    title : str, optional
        Title for the plots. By default 'Colormaps'.
    """

    #Allow a Single Cmap Name Without a List
    if not isinstance(cmap_names_list, list):
        cmap_names_list = [ cmap_names_list ]

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_names_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    
    #Set Title
    axs[0].set_title(f'{title}', fontsize=14)

    predefined_colors = get_predefined_colors()

    #Create Sample of Colormap
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack([gradient, gradient])


    for ax, name in zip(axs, cmap_names_list):

        #Set Colormap
        if name in predefined_colors:
            colormap = custom_cmap(name)
        else:
            colormap = mpl.colormaps[name]
        
        #Plot
        ax.imshow(gradient, aspect='auto', cmap= colormap)

        #Label
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()




######################     Covmat     ############################################
def plotCovmat(Covmat, covmat_labels,
               Cls_names_to_plot = None,
               slice_Cls = None,
               clim = None,
               figure_size = (10,10)
              ):
    """
    Plot a covmat.

    Parameters
    ----------
    Covmat : 2darray
        Covmat to plot. Can be an individual covmat or a larger, multi-probe covmat.
    covmat_labels : list
        Names of either fields or Cl's that describe the "big_covmat" (see "labels" argument of "getCovmatInfo" for more info).
    Cls_names_to_plot : list, optional
        List of spectra whose names you want (see "select_Cls_list" argument of util.selectIndivCovmats for more info). By default None.
    slice_Cls : list, optional
        Names of spectra used for slicing the covmat. Format is [start_Cl1, start_Cl2, end_Cl1, end_Cl2] (see arguments of the same names from util.sliceCovmat for more info). By default None.
    clim : list, optional
        List of values in covmat that correspond to the colorbar min and max. Format: [min, max]. By default None.
    figure_size : tuple, optional
        Size of the output plot. By default (10,10).

    Returns
    -------
    fig, ax
        Figure and axis of the output plot.

    Raises
    ------
    ValueError
        Can only provide one of the following: 'Cls_names_to_plot', 'slice_Cls'  
    """    
    if Cls_names_to_plot is not None and slice_Cls is not None: 
        raise ValueError("Specifying both 'Cls_names_to_plot' and 'slice_Cls' is ambiguous. Please provide only one.")
    
    #Get Spectra Names Corresponding to Covmat
    if Cls_names_to_plot is  None:
        Cls_names = yutil.getCovmatInfo(covmat_labels, 'cls')

    #Use Custom Spectra
    else:
        Cls_names = Cls_names_to_plot

    #Set Slice Covmat Defaults
    if slice_Cls is None:
        subCovmat_indices = [ 0, 0, len(Cls_names)-1, len(Cls_names)-1 ]

    #Slice Covmat
    else:
        _, subCovmat_indices = yutil.sliceCovmat(*slice_Cls, Covmat, covmat_labels, return_indices= True)

    #Colors
    if clim is None:
        cmin = Covmat.min()
        cmax = Covmat.max()
    else:
        cmin = clim[0]
        cmax = clim[1]

    #Set up Figure
    N_iax = len( np.arange(subCovmat_indices[0], subCovmat_indices[2]+1) )
    N_jax = len( np.arange(subCovmat_indices[1], subCovmat_indices[3]+1) )

    fig, ax = plt.subplots( N_iax, N_jax, 
                            figsize = figure_size,
                            sharex = True,
                            sharey = True,
                            squeeze= False)
    
    #Plot Individual Covmats on the Large Covmat Canvas
    iax = 0
    for iCov, Cl_name_row in enumerate(Cls_names):

        if iCov not in range(subCovmat_indices[0], subCovmat_indices[2]+1):
            continue

        jax = 0
        for jCov, Cl_name_col in enumerate(Cls_names):

            if jCov not in range(subCovmat_indices[1], subCovmat_indices[3]+1):
               continue

            #Get Individual Covmat
            indiv_covmat = yutil.getIndividualCovmat(Cl_name_row, Cl_name_col, Covmat, covmat_labels)

            #Plot Individual Covmat
            ax[iax,jax].imshow(indiv_covmat, cmap='RdBu', vmin= cmin, vmax= cmax)

            #Adjust Individual Covmat's Axis
            ax[iax,jax].set_aspect('equal')
            ax[iax,jax].tick_params(axis='both', 
                                direction='in',
                                top= True, right= True,
                                labelbottom= False, labelleft= False)

            #Create Latex Label
            ifields = Cl_name_row.split('x')
            for idx, field in enumerate(ifields):
                if field == 'kappa':
                    ifields[idx] = '\kappa'
            Cl_name_row_latex = '\;\mathrm{x}\;'.join( ifields ) 
            jfields = Cl_name_col.split('x')
            for idx, field in enumerate(jfields):
                if field == 'kappa':
                    jfields[idx] = '\kappa'
            Cl_name_col_latex = '\;\mathrm{x}\;'.join( jfields ) 

            #Add Cl Labels
            ax[iax,jax].xaxis.set_label_position('top')
            if jax == 0:
                ax[iax,jax].set_ylabel(fr'${Cl_name_row_latex}$', size='x-large')
            if iax == 0:
                ax[iax,jax].set_xlabel(fr'${Cl_name_col_latex}$', size='x-large')

            jax += 1

        iax += 1

    plt.subplots_adjust(wspace=0, hspace=0, 
                        right=0.9,
                        bottom=0.1)

    #Colorbar
    cbar_ax = fig.add_axes([0.95, 0.10, 0.03, 0.8])
    norm = mpl.colors.Normalize(vmin= cmin, vmax= cmax)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu'),
                #  ax = ax.ravel().tolist() 
                 cax = cbar_ax)

    return fig, ax



def savefig(fname, figure, **kwargs):
    #Default Options
    default_options = {}
    default_options['bbox_inches'] = 'tight'
    default_options['dpi'] = 500

    options = { **default_options, **kwargs }
    figure.savefig(fname, **options)



def set_rcParams(**kwargs):
    #Default Changes
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.figsize'] = [8, 6]

    #Update With Custom Parameter Changes
    mpl.rcParams.update(kwargs)



def ratioPlot(xs= None, 
              yref_dict={},
              ylines_dict={},
              ref_kwargs={},
              lines_kwargs={},
              ratio_type = 'percent discrepancy'
             ):

    valid_ratios = ['percent discrepancy', 'ratio']
    if ratio_type not in valid_ratios:
        raise ValueError(f"The only valid ratios are {valid_ratios}")

    #Setup
    set_rcParams()
    fig, ax = plt.subplots(1,2, figsize=(15,5))

    #Plot Data
    if xs is not None:
        #Absolute Plot
        for ylabel, yline in ylines_dict.items():
            ax[0].plot(xs, yline, label= ylabel, **lines_kwargs)
        for ref_label, ref_line in yref_dict.items():
            ax[0].plot(xs, ref_line, color='black', label= ref_label, **ref_kwargs)

        #Relative Plot
        for ylabel, yline in ylines_dict.items():
            if ratio_type == 'percent discrepancy':
                ax[1].plot(xs, ( yline - ref_line) / ref_line  * 100, label= ylabel, **lines_kwargs)
            elif ratio_type == 'ratio':
                ax[1].plot(xs,  yline / ref_line  * 100, label= ylabel, **lines_kwargs)
    
    #Plot Horizontal Line on Relative Plot
    if ylines_dict:
        if ratio_type == 'percent discrepancy':
            ax[1].axhline(y= 0, color='black', label= ref_label)
        elif ratio_type == 'ratio':
            ax[1].axhline(y= 1 * 100, color='black', label= ref_label)

    #Trimmings: Absolute Plot
    ax[0].set_ylabel(r'$C_{\ell}$')
    ax[0].set_xlabel(r'$\ell$')
    ax[0].grid()
    ax[0].legend()

    #Trimmings: Relative Plot
    if ratio_type == 'percent discrepancy':
        ax[1].set_title('Percent Discrepancy')
        ax[1].set_ylabel('[%]')
        if xs is not None:
            ax[1].set_ylabel(rf'$(C_{{\ell}} - C_{{\ell}}^{{ {list(yref_dict)[0]} }} )/ C_{{\ell}}^{{ {list(yref_dict)[0]} }} \; [\%]$')
    elif ratio_type == 'ratio':
        ax[1].set_title('Ratios')
        ax[1].set_ylabel('[%]')
        if xs is not None:
            # ax[1].set_ylabel(rf'$C_{{\ell}} / C_{{\ell}}^{{ \textrm{{ {list(yref_dict)[0]} }} }} \; [\%]$')
            ax[1].set_ylabel(rf'$C_{{\ell}} / C_{{\ell}}^{{ {list(yref_dict)[0]} }} \; [\%]$')
    ax[1].set_xlabel(r'$\ell$')
    ax[1].grid(which='both')
    ax[1].tick_params(which='both')
    ax[1].legend()

    return fig, ax