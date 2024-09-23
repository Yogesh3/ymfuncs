import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')
from pixell import enplot, colorize
import numpy as np
from myfuncs import util as yutil


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