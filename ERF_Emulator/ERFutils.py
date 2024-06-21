import xarray as xr
import pandas as pd
import numpy as np
import cftime
import dask
import xarrayutils
import cartopy.crs as ccrs
import xesmf as xe
import scipy.signal as signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve_triangular
import math

import os
import copy
import seaborn as sns
import matplotlib as mpl
import cmocean
import cmocean.cm as cmo

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

########################### Matplotlib Definitions ###################################

matplotlib.rcdefaults()
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

########################### Path Definitions ###################################

path_to_cmip6_data_local = 'path/to/cmip6_data/'
path_to_cmip6_data = 'path/to/remote/cmip6_data/'
path_to_piControl = f'{path_to_cmip6_data}piControl/'
path_to_ERF_outputs = '../ERF_Outputs/'
path_to_figures = f'{path_to_ERF_outputs}Figures/'

########################### Import and Regrid Data ###################################

def check_imported(experiments,models,variables,print_path=False):
    """
    Checks if the listed experiments, models, and variables are downloaded.
    Note: only checks if the variables exist, does not currently have a way
    to check if the complete dataset is downloaded. Assumes if a variable
    exists the entire dataset was downloaded correctly.
    
    Args:
        experiments: List of experiments to check.
        models: Set of models to check.
        variables: List of variables to check.
        print_path: Whether or not to print the output path for debugging.
        
    Returns.
        None.
    """
    path_to_cmip6_data = path_to_cmip6_data_local
    failed = {}
    
    # Iterate over experiments and check if they can be loaded by xarray
    for exp_id in experiments:
        print(f'Experiment: {exp_id}') 
        failed[exp_id] = {}
        for m in models:
            print(f'\tModel: {m}')
            failed[exp_id][m] = []
            for var in variables:
                try:
                    if print_path:
                        print(f'{path_to_cmip6_data}{exp_id}/{var}_Amon_{m}_{exp_id}_r1i1p1f1**')
                    xr.open_mfdataset(f'{path_to_cmip6_data}{exp_id}/{var}_Amon_{m}_{exp_id}_r1i1p1f1**', use_cftime=True)
                except:
                    failed[exp_id][m].append(var)
                    
            if len(failed[exp_id][m]) == 0:
                print('\t\tAll variables present!')
            else:
                print(f'\t\tMissing: {[m_var for m_var in failed[exp_id][m]]}')
                
    return

def load_regrid_ERF_set(model_set, data_id, verbose = False):
    """
    Loads ERF for a given dataset.
    NOTE: This function is rather slow and can likely be optimized.
    
    Args:
        model_set: Set of model names within the given dataset.
        data_id: ID of the dataset, e.g. 1pctCO2.
        print_var: Boolean to print variable path, used for debugging.
        
    Returns:
        ERF: Dictionary organized by model containing ERF data.
        dN: Dictionary organized by model containing raw radiative flux data.
        EB_train_dict: Dictionary organized by model containing energy balance data for the training dataset.
        EB_ctrl_dict: Dictionary organized by model containing energy balance data for the control dataset.
    """

    lam = lam_dict
    to_load = ['rsdt','rsut','rlut','tas']
    ERF, EB_dict = {}, {}
    
    # Iterate over models
    for m in model_set:
        print(f'\t Loading {m} data...')
        EB_dict[m] = {}
        
        # Iterate over fluxes and load each variable
        for var in to_load:
            if verbose:
                print(f'{path_to_cmip6_data}{data_id}/{var}_Amon_{m}_{data_id}_r1i1p1f1....nc')
                print(f'{path_to_cmip6_data}piControl/{var}_Amon_{m}_piControl_r1i1p1f1....nc')
                
            # Load variable and remove climatology
            EB_dict[m][var] = remove_climatology(data_id, m, var)

        # Calculate energy balance
        EB_dict[m]['RF'] = calc_energy_balance(EB_dict[m]['rlut']['rlut'], EB_dict[m]['rsdt']['rsdt'], EB_dict[m]['rsut']['rsut'])    
        ERF[m] = (EB_dict[m]['RF'].RF + lam[m]*EB_dict[m]['tas'].tas).to_dataset(name = 'ERF')
        
        # Start from t = 0, such that all model times match
        ERF[m]['year'] = range(len(ERF[m]['year']))
    
        # Regrid
        ERF[m] = regrid_cmip(ERF[m], ds_out)
        
        # Drop height, unecessary
        ERF[m] = ERF[m].drop_vars('height')
    
    # Concatenate models into a single dataset
    ERF = concat_multirun(ERF,'model')
    
    return ERF

def load_regrid_tas_set(model_set, data_id):
    """
    Loads tas for a given dataset.
    NOTE: This function is rather slow and can likely be optimized.
    
    Args:
        model_set: Set of model names within the given dataset.
        data_id: ID of the dataset, e.g. 1pctCO2.
        
    Returns:
        tas_ds: Dataset containing tas data.
    """
    
    tas = {}
    min_year = 100000
    for m in model_set:     
        print(f'\t Loading {m} data...')
        # Load tas and remove climatology
        tas[m] = remove_climatology(data_id, m, 'tas')

        # Start from t = 0, such that all model times match
        if len(tas[m]['year']) < min_year:
            min_year = len(tas[m]['year'])
        tas[m]['year'] = range(len(tas[m]['year']))

        # Regrid
        tas[m] = regrid_cmip(tas[m], ds_out)
    
    # Concatenate models into a single dataset
    tas_ds = concat_multirun(tas,'model')
    tas_ds = tas_ds.rename({'year':'s'})
    
    # Drop height as it is unecessary
    tas_ds = tas_ds.drop_vars('height')
    
    # Ensure all models are aligned in time
    cropped_data = []
    models = []
    for m in model_set:
        cropped_data.append(tas_ds.sel(model=m)['tas'].isel(s=slice(0, min_year)))
        models.append(m)
    
    # Ensure all cropped dimensions align properly
    cropped_ds = xr.Dataset(coords={'lon': ('lon', tas_ds.lon.values),
                                    'lat': ('lat', tas_ds.lat.values),
                                    's': ('s', range(min_year)),
                                    'model': ('model', tas_ds.model.values)})
    cropped_ds = cropped_ds.assign(tas=(['model','s','lat','lon'],cropped_data))
    cropped_ds['model'] = models
    
    return cropped_ds


def calc_climatology(m, var):
    """
    Get the average climatology for a given variable and model.
    
    Args:
        m: Model name.
        var: Variable to get climatology for.
        
    Returns:
        piControl_ds: Dataset containing spatially resolved climatology data.
    """
    
    # Load and convert piControl data to yearly
    piControl_ds = xr.open_mfdataset(f'{path_to_cmip6_data}piControl/{var}_Amon_{m}_piControl_r1i1p1f1**.nc', use_cftime=True, chunks = 1000000)
    piControl_ds = monthly_to_annual(piControl_ds)
    
    # Average piControl over the entire simulation period
    piControl_ds = piControl_ds.mean(dim = ['year'])
    
    return piControl_ds.as_numpy()

def remove_climatology(data_id, m, var):
    """
    Remove the average climatology for a given variable and model.
    
    Args:
        data_id: Dataset to remove climatology from.
        m: Model name.
        var: Variable to get climatology for.
        
    Returns:
        ds_rem: Dataset containing spatially resolved climatology data.
    """
    
    # Load and convert dataset to yearly
    ds = xr.open_mfdataset(f'{path_to_cmip6_data}{data_id}/{var}_Amon_{m}_{data_id}_r1i1p1f1**.nc', use_cftime=True, chunks = 1000000)
    ds = monthly_to_annual(ds).as_numpy()
    
    # Calculate climatology
    climatology = calc_climatology(m, var)
    
    # Remove climatology from dataset
    ds_rem = (ds[var] - climatology[var]).to_dataset(name = var)
    
    return ds_rem

def regrid_cmip(ds, ds_out):
    """
    Regrid a dataset to our chosen output lat/lon size.
    
    Args:
        ds: Data to regrid.
        ds_out: Size of the grid you want, must have lat, lon, lat_b, lon_b.
        
    Returns:
        ds: Regridded dataset.
    """
    
    attrs = ds.attrs
    if False:
        regridder = xe.Regridder(ds, ds_out, "conservative")
        ds = regridder(ds) 
    else:
        regridder = xe.Regridder(get_bounds(ds, 1.0), get_bounds(ds_out, 1.0), 'conservative')
        ds = regridder(ds)
    
    ds.attrs = attrs
    
    return ds

def get_bounds(arr, gridSize):    
    lonMin = np.nanmin(arr["lon"].values)
    latMin = np.nanmin(arr["lat"].values)
    lonMax = np.nanmax(arr["lon"].values)
    latMax = np.nanmax(arr["lat"].values)
    
    sizeLon = len(arr["lon"])
    sizeLat = len(arr["lat"])
    
    bounds = {}
    
    bounds["lon"] = arr["lon"].values
    bounds["lat"] = arr["lat"].values
    bounds["lon_b"] = np.linspace(lonMin-(gridSize/2), lonMax+(gridSize/2), sizeLon+1)
    bounds["lat_b"] = np.linspace(latMin-(gridSize/2), latMax+(gridSize/2), sizeLat+1).clip(-90, 90)
    
    return bounds

########################### ERF Specific Functions ###################################

def calc_energy_balance(rlut, rsdt, rsut):
    """
    Calculate energy balance for a given dataset.
    
    Args:
        rlut: Outgoing longwave radiation.
        rsdt: Incoming shortwave radiation.
        rsut: Outgoing shortwave radiation.
    
    Returns:
        EB: Dataset containing radiative flux.
    """
    longwave_balance = -rlut
    shortwave_balance = rsdt - rsut
    EB = (longwave_balance + shortwave_balance).to_dataset(name = 'RF')
    
    return EB

def calc_ERF_sensitivity(ERF_ds,tas_ds,model_set,lam_dict,pct_sens):
    """
    Modifies ERF profile in order to perform sensitivity analysis.
    Assumes an existing ERF dataset, does not calculate energy balance
    from scratch.
    
    Args:
        ERF_ds: Original ERF dataset.
        tas_ds: Original tas dataset.
        model_set: Set containing all models to iterate over.
        lam_dict: Dictionary mapping models to their feedback parameters.
        pct_sens: Percentage by which to modify the feedback parameter.
    
    Returns:
        ERF_sens: Modified ERF dataset.
    """
    ERF_sens = copy.deepcopy(ERF_ds)

    for m in model_set:
        tas_glob_m_ds = tas_ds.sel(model=m).weighted(A).mean(dim=['lat','lon']).tas.values
        ERF_m = ERF_ds.sel(model=m).ERF.values
        lam = lam_dict[m]
        dN = ERF_m - lam*tas_glob_m_ds 
        
        ERF_sens_m = dN + (1 + pct_sens)*lam*tas_glob_m_ds
        ERF_sens.ERF.loc[dict(model=m)] = ERF_sens_m
        
    return ERF_sens

########################### Diagnose Green's Functions and Patterns ###################################

def diagnose_mean_GF(train, plot = True, save_data = False, save_fig = False):
    """
    Diagnoses multimodel mean GF from a given experiment. Assumes ERF and tas
    datasets are formatted properly. Smoothing parameters are tuned relative
    to the 1pctCO2 experiment.
    
    Args:
        train: Dataset to use for diagnosis.
        plot: Whether or not to plot the resulting Green's function
        save_data: Whether or not to save the resulting Green's function
        save_fig: Whether or not to save the plotted GF
        
    Returns:
        G_ds: Dataset containing multimodel mean Green's function
    """
    
    # Load ERF data
    ERF = {}
    ERF_path = f'{path_to_ERF_outputs}ERF/ERF_{train}_ds.nc4'
    ERF_ds = xr.open_dataset(ERF_path)
    ERF[train] = ds_to_dict(ERF_ds)
    ERF_all = concat_multirun(ERF[train],'model').mean(dim = 'model')

    # Load tas data
    tas_path = f'{path_to_ERF_outputs}tas/tas_{train}_ds.nc4'
    tas_ds = xr.open_dataset(tas_path)
    tas_ds = tas_ds.rename({'s': 'year'})

    # Ensure years align
    tas_ds = tas_ds.sel(year = slice(ERF_all['year'].min(), ERF_all['year'].max()))

    # Organization for data smoothing
    X1 = ERF_all.year.values
    Y1 = ERF_all.ERF.values
    X2 = tas_ds.year.values

    # Smoothing parameters, tuned manually for 1pctCO2 experiment
    tau = 20
    j = 4
    N_years = len(ERF_all['year']) - j
    offsets = [i for i in range(0,-N_years,-1)]
    domain = np.linspace(ERF_all.year.values[j], ERF_all.year.values[-1], num=N_years)

    # Stack tas to vectorize smoothing operation
    tas_stack = tas_ds.stack(allpoints=['lat','lon']).mean(dim = ['model'])
    tas_stack_vals = tas_stack.tas.values

    # Smooth ERF and tas data
    ERF_smooth = [local_weighted_regression(x0, X1, Y1, tau) for x0 in domain]
    tas_smooth = [local_weighted_regression(x0, X2, tas_stack_vals, tau) for x0 in domain]
    input_matrix = diags(ERF_smooth,offsets=offsets,shape=(N_years,N_years),format='csr')
    array_mat = input_matrix.toarray()

    # Solve for G
    G_stack = spsolve_triangular(input_matrix,tas_smooth,lower=True)

    # Reformat G
    G_ds = xr.Dataset(coords={'lon': ('lon', tas_ds.lon.values),
                                'lat': ('lat', tas_ds.lat.values),
                                'year': ('year', range(N_years))})
    G_ds = G_ds.stack(allpoints=['lat','lon'])
    G_ds['G[tas]'] = (('year','allpoints'),G_stack)
    G_ds = G_ds.unstack('allpoints')
    G_ds['year'] = G_ds['year'] - G_ds['year'][0]
    G_ds = G_ds.weighted(A).mean(dim = ['lat','lon'])['G[tas]']
    
    # Plot resultant Green's function
    if plot:
        fig, ax = plt.subplots(figsize = [8,6])
        ax.plot(G_ds.weighted(A).mean(dim = ['lat','lon']).values[0:31], linewidth=3, color=brewer2_light(2))
        ax.set_title(f'Global Average Green\'s Function: 1pctCO2 Experiment',fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel('Years Since ERF Impulse',fontsize=20)
        ax.set_ylabel('Impulse Response [$^\circ$C/(Wm$^{-2}$)]',fontsize=20)
        plt.grid(True)
        fig.tight_layout()
        
        if save_fig:
            plt.savefig(f'{path_to_figures}global_GF_{train}.pdf', bbox_inches = 'tight', dpi = 500)

    # Save resultant Green's function
    if save_data:
        G_ds.to_netcdf(f'{path_to_ERF_outputs}G_{train}_mean_ds.nc4')
    
    return G_ds

def diagnose_mean_pattern(train, plot = True, save_data = False, save_fig = False):
    """
    Diagnoses multimodel mean pattern from a given experiment. Assumes tas dataset
    is formatted properly.
    
    Args:
        train: Dataset to use for diagnosis.
        plot: Whether or not to plot the resulting pattern
        save_data: Whether or not to save the resulting pattern
        save_fig: Whether or not to save the plotted pattern
        
    Returns:
        pattern_ds: Dataset containing multimodel mean pattern
    """
    # Load tas data
    tas_path = f'{path_to_ERF_outputs}tas/tas_{train}_ds.nc4'
    tas_ds = xr.open_dataset(tas_path)
    tas_glob_mean = tas_ds.weighted(A).mean(dim = ['lat','lon']).mean(dim = ['model'])
    tas_glob_mean = tas_glob_mean.rename({'s': 'year'})

    # Stack tas to vectorize smoothing operation
    tas_stack = tas_ds.stack(allpoints=['lat','lon']).mean(dim = ['model'])
    tas_stack_vals = tas_stack.tas.values

    # Get size of resulting pattern
    N_latlong = len(tas_stack['allpoints'].values)

    # Reshape tas_glob_mean for matrix operations
    X = tas_glob_mean.tas.values.reshape(-1, 1)  # shape (n_samples, 1)

    # Prepare Y for batch processing
    Y = tas_stack_vals  # shape (n_samples, N_latlong)

    # Compute the pattern using the normal equation
    # (X^T X)^-1 X^T Y
    XTX_inv = np.linalg.inv(X.T @ X)  # shape (1, 1)
    XTY = X.T @ Y  # shape (1, N_latlong)
    pattern_stacked = (XTX_inv @ XTY).reshape(1, N_latlong)  # shape (1, N_latlong)

    # Reformat pattern
    pattern_ds = xr.Dataset(coords={'lon': ('lon', tas_ds.lon.values),
                            'lat': ('lat', tas_ds.lat.values)})
    pattern_ds = pattern_ds.stack(allpoints=['lat','lon'])
    pattern_ds['pattern'] = ('allpoints',pattern_stacked[0])
    pattern_ds = pattern_ds.unstack('allpoints')
    
    # Plot resultant pattern
    if plot:
        plot_pattern(pattern_ds,train,test_id = None,save_fig = False)

        if save_fig:
            plt.savefig(f'{path_to_figures}global_pattern_{train}.pdf', bbox_inches = 'tight', dpi = 500)

    # Save resultant pattern
    if save_data:
        pattern.to_netcdf(f'{path_to_ERF_outputs}pattern_{train}_mean_ds.nc4')
    
    return pattern_ds

########################### Convolution and Analysis ###################################

def eval_GF(train_id, conv_all, plot, save_result, save_fig, verbose = True, sens = False, pct_sens = None):
    """
    Evaluate the performance of a Green's function in terms of globally averaged
    RMSE, MAE, bias, and relative bias.
    
    Args:
        train_id: Green's function chosen for evaluation.
        conv_all: List of experiments used for convolution.
        verbose: Whether or not to print the results.
        sens: Flag to indicate a sensitivity analysis.
        pct_sens: Percentage (plus and minus) by which to modify the 
                  climate feedback parameter.
        
    Returns:
        RMSE_short: Short term RMSE results.
        RMSE_long: Long term RMSE results.
        MAE_short: Short term MAE results.
        MAE_long: Long term MAE results.
        bias_short: Short term bias results.
        bias_long: Long term bias results.
        rel_bias_short: Short term relative bias results.
        long_bias_short: Long term relative bias results.
    """

    # Import Green's functions
    G_ds_path = f'{path_to_ERF_outputs}GFs/G_{train_id}_mean_ds.nc4'
    G_ds = xr.open_dataset(G_ds_path)['G[tas]']
    G_ds.name = 'G[tas]'
    G_ds = G_ds.rename({'year':'s'})

    # Arrays for storing error statistics
    RMSE_short, RMSE_long = [], []
    MAE_short, MAE_long = [], []
    bias_short, bias_long = [], []
    rel_bias_short, rel_bias_long = [], []
    
    # Arrays for storing sensitivity analysis
    if sens:
        RMSE_short_p10, RMSE_long_p10, RMSE_short_m10, RMSE_long_m10 = [], [], [], []
        MAE_short_p10, MAE_long_p10, MAE_short_m10, MAE_long_m10 = [], [], [], []
        bias_short_p10, bias_long_p10, bias_short_m10, bias_long_m10 = [], [], [], []
        rel_bias_short_p10, rel_bias_long_p10, rel_bias_short_m10, rel_bias_long_m10 = [], [], [], []
    
    for conv_id in conv_all:
        print(f'\tLoading {conv_id} experiment for convolution...')

        # Import experimental ERF data
        ERF_path = f'{path_to_ERF_outputs}ERF/ERF_{conv_id}_ds.nc4'
        if 'ssp' in conv_id:
            ERF_path_hist = f'{path_to_ERF_outputs}ERF/ERF_historical_ds.nc4'
            ERF_ssp = xr.open_dataset(ERF_path)
            ERF_hist = xr.open_dataset(ERF_path_hist)

            ERF_ds = xr.concat([ERF_hist,ERF_ssp.assign_coords(year = range(165,250))],dim = 'year')

        else:
            ERF_ds = xr.open_dataset(ERF_path)

        # Import experimental tas data
        tas_path = f'{path_to_ERF_outputs}tas/tas_{conv_id}_ds.nc4'
        tas_ds = xr.open_dataset(tas_path)
        
        # tas_ds has a mismatch in the length between models
        if '1pctCO2' in conv_id:
            tas_ds = tas_ds.sel(s=slice(0,149))

        # Convolve ERF profile with Green's functions
        conv_mean_ds = convolve_exp_meanGF(G_ds, ERF_ds, train_id, conv_mean = True)
        conv_ds = convolve_exp_meanGF(G_ds, ERF_ds, train_id, conv_mean = False)
        
        # For sensitivity analysis, recalculate ERF values and perform new convolutions
        if sens:
            ERF_m10_ds = calc_ERF_sensitivity(ERF_ds,tas_ds,model_set,lam_dict,-0.1)
            ERF_p10_ds = calc_ERF_sensitivity(ERF_ds,tas_ds,model_set,lam_dict,0.1)
            
            conv_mean_m10_ds = convolve_exp_meanGF(G_ds, ERF_m10_ds, train_id, conv_mean = True)
            conv_m10_ds = convolve_exp_meanGF(G_ds, ERF_m10_ds, train_id, conv_mean = False)
            conv_mean_p10_ds = convolve_exp_meanGF(G_ds, ERF_p10_ds, train_id, conv_mean = True)
            conv_p10_ds = convolve_exp_meanGF(G_ds, ERF_p10_ds, train_id, conv_mean = False) 

        # Save convolution outputs
        if save_result:
            conv_mean_ds.to_netcdf(f'{path_to_ERF_outputs}Global Mean Results/res_conv_global_{train_id}_{conv_id}_ds.nc4')
            conv_ds.to_netcdf(f'{path_to_ERF_outputs}Spatial Results/res_conv_spatial_{train_id}_{conv_id}_ds.nc4')
            
            if sens:
                conv_mean_m10_ds.to_netcdf(f'{path_to_ERF_outputs}Global Mean Results/res_conv_global_{train_id}_{conv_id}_m10_ds.nc4')
                conv_m10_ds.to_netcdf(f'{path_to_ERF_outputs}Spatial Results/res_conv_spatial_{train_id}_{conv_id}_m10_ds.nc4')
                conv_mean_p10_ds.to_netcdf(f'{path_to_ERF_outputs}Global Mean Results/res_conv_global_{train_id}_{conv_id}_p10_ds.nc4')
                conv_p10_ds.to_netcdf(f'{path_to_ERF_outputs}Spatial Results/res_conv_spatial_{train_id}_{conv_id}_p10_ds.nc4')

        # Select short and long term time periods based on experiment
        if 'ssp' in conv_id: # SSP experiments
            start_yr1, plot_yr1, end_yr1 = 2040, 2050, 2060
            start_yr2, plot_yr2, end_yr2 = 2080, 2090, 2100
        elif 'hist' in conv_id: # Historical experiment
            start_yr1, plot_yr1, end_yr1 = 1900, 1920, 1940
            start_yr2, plot_yr2, end_yr2 = 1985, 2000, 2015
        else: # 1pctCO2 experiment
            start_yr1, plot_yr1, end_yr1 = 1940, 1950, 1960
            start_yr2, plot_yr2, end_yr2 = 1980, 1990, 2000
        
        # Make plots for analysis, comment out lines depending on desired plots
        if plot:
            if sens:
                plot_conv_meanGF(train_id, conv_id, conv_mean_ds, tas_ds, sens = True,
                              conv_mean_p10_ds = conv_mean_p10_ds, conv_mean_m10_ds = conv_mean_m10_ds,
                              save_fig = save_fig)
            else:
                plot_ERF_profile(ERF_ds, conv_id, save_fig = False)
                plot_conv_meanGF(train_id, conv_id, conv_mean_ds, tas_ds, save_fig = save_fig)
                plot_dif_map_meanGF(conv_ds, tas_ds, plot_yr = plot_yr1, yr_dif = 10, conv_id = conv_id, train_id = train_id, dif = True, save_fig = save_fig)
                plot_dif_map_meanGF(conv_ds, tas_ds, plot_yr = plot_yr2, yr_dif = 10, conv_id = conv_id, train_id = train_id, dif = True, save_fig = save_fig)

        # Calculate and record error values
        MSE1, RMSE1, MAE1, bias1, rel_bias1 = calc_error_metrics(tas_ds, conv_ds, start_yr1, end_yr1, mean_GF = False)
        RMSE_short.append(round(RMSE1,4))
        MAE_short.append(round(float(MAE1),4))
        bias_short.append(round(float(bias1),4))
        rel_bias_short.append(round(float(rel_bias1),4))

        MSE2, RMSE2, MAE2, bias2, rel_bias2 = calc_error_metrics(tas_ds, conv_ds, start_yr2, end_yr2, mean_GF = False)
        RMSE_long.append(round(RMSE2,4))
        MAE_long.append(round(float(MAE2),4))
        bias_long.append(round(float(bias2),4))
        rel_bias_long.append(round(float(rel_bias2),4))
        
        if sens:
            MSE1_p10, RMSE1_p10, MAE1_p10, bias1_p10, rel_bias1_p10 = calc_error_metrics(tas_ds, conv_p10_ds, start_yr1, end_yr1, mean_GF = False)
            RMSE_short_p10.append(round(RMSE1_p10,4))
            MAE_short_p10.append(round(float(MAE1_p10),4))
            bias_short_p10.append(round(float(bias1_p10),4))
            rel_bias_short_p10.append(round(float(rel_bias1_p10),4))

            MSE1_m10, RMSE1_m10, MAE1_m10, bias1_m10, rel_bias1_m10 = calc_error_metrics(tas_ds, conv_m10_ds, start_yr1, end_yr1, mean_GF = False)
            RMSE_short_m10.append(round(RMSE1_m10,4))
            MAE_short_m10.append(round(float(MAE1_m10),4))
            bias_short_m10.append(round(float(bias1_m10),4))
            rel_bias_short_m10.append(round(float(rel_bias1_m10),4))
            
            MSE2_p10, RMSE2_p10, MAE2_p10, bias2_p10, rel_bias2_p10 = calc_error_metrics(tas_ds, conv_p10_ds, start_yr2, end_yr2, mean_GF = False)
            RMSE_long_p10.append(round(RMSE2_p10,4))
            MAE_long_p10.append(round(float(MAE2_p10),4))
            bias_long_p10.append(round(float(bias2_p10),4))
            rel_bias_long_p10.append(round(float(rel_bias2_p10),4))
        
            MSE2_m10, RMSE2_m10, MAE2_m10, bias2_m10, rel_bias2_m10 = calc_error_metrics(tas_ds, conv_m10_ds, start_yr2, end_yr2, mean_GF = False)
            RMSE_long_m10.append(round(RMSE2_m10,4))
            MAE_long_m10.append(round(float(MAE2_m10),4))
            bias_long_m10.append(round(float(bias2_m10),4)) 
            rel_bias_long_m10.append(round(float(rel_bias2_m10),4))
            
    
    # Print results
    if verbose:
        print('\nResults are shown in the following order:')
        print(conv_all)

        print('\nMid-Century Stats:')
        print(f'RMSE: {RMSE_short}')
        print(f'MAE: {MAE_short}')
        print(f'Bias: {bias_short}')
        print(f'Relative Bias: {rel_bias_short}')

        print('\nEnd-of-Century Stats:')
        print(f'RMSE: {RMSE_long}')
        print(f'MAE: {MAE_long}')
        print(f'Bias: {bias_long}')
        print(f'Relative Bias: {rel_bias_long}')
        
        if sens:
            print('\nMid-Century Stats +10%:')
            print(RMSE_short_p10)
            print(MAE_short_p10)
            print(bias_short_p10)
            print(rel_bias_short_p10)

            print('\nMid-Century Stats -10%:')
            print(RMSE_short_m10)
            print(MAE_short_m10)
            print(bias_short_m10)
            print(rel_bias_short_m10)

            print('\nEnd-of-Century Stats +10%:')
            print(RMSE_long_p10)
            print(MAE_long_p10)
            print(bias_long_p10)
            print(rel_bias_long_p10)

            print('\nEnd-of-Century Stats -10%:')
            print(RMSE_long_m10)
            print(MAE_long_m10)
            print(bias_long_m10)
            print(rel_bias_long_m10)
            
    if sens:
        return (RMSE_short, RMSE_long, RMSE_short_p10, RMSE_long_p10,
                RMSE_short_m10, RMSE_long_m10, MAE_short, MAE_long,
                MAE_short_p10, MAE_long_p10, MAE_short_m10, MAE_long_m10,
                bias_short, bias_long, bias_short_p10, bias_long_p10,
                bias_short_m10, bias_long_m10, rel_bias_short, rel_bias_long,
                rel_bias_short_p10, rel_bias_long_p10, rel_bias_short_m10, rel_bias_long_m10)
    else:
        return RMSE_short, RMSE_long, MAE_short, MAE_long, bias_short, bias_long, rel_bias_short, rel_bias_long

def convolve_exp_meanGF(G_ds, ERF_ds, train_id, conv_mean = True):
    """
    Convolves a given experiment ERF profile with a Green's function
    to get the temperature response.
    
    Args:
        G_ds: Green's function dataset.
        ERF_ds: ERF dataset.
        train_id: ID indicating training dataset.
        conv_mean: Convolve with the global mean or all locations globally.
        
    Returns:
        conv_ds: Dataset containing convolved temperature response.
    """
    
    # Set to convolve with the global mean GF
    if conv_mean:
        G_ds = G_ds.weighted(A).mean(dim = ['lat','lon'])

    GF = G_ds
    ERF_ds = ERF_ds.mean(dim = ['model'])
        
    conv = {} 
    if conv_mean:
        print(f'Convolving mean GF for Global Mean')
        conv[train_id] = signal.convolve(np.array(GF.dropna(dim = 's')),
                                            np.array(ERF_ds['ERF']),'full')
        conv[train_id] = np_to_xr_mean(conv[train_id], GF, ERF_ds)

    else:
        print(f'Convolving mean GF Spatially')
        conv[train_id] = signal.convolve(np.array(GF.dropna(dim = 's')), 
                                       np.array(ERF_ds['ERF'])[~np.isnan(np.array(ERF_ds['ERF']))][..., None, None],
                                       'full')
        conv[train_id] = np_to_xr(conv[train_id], GF, ERF_ds)
    
    # Ensure length of convolution is correct
    length = max(len(GF.dropna(dim = 's')['s']),len(np.array(ERF_ds['ERF'])))
    conv[train_id] = conv[train_id][:length]
    
    # Reformat dataset
    conv_ds = concat_multirun(conv,'train_id')

    return conv_ds

def scale_pattern(train_id, scale_all, plot, save_result, save_fig, verbose = True):
    """
    Evaluate the performance of pattern scaling in terms of globally averaged
    RMSE, MAE, and relative bias.
    
    Args:
        train_id: Pattern chosen for evaluation.
        scale_all: List of experiments used for pattern scaling.
        verbose: Whether or not to print the results.

    Returns:
        RMSE_short: Short term RMSE results.
        RMSE_long: Long term RMSE results.
        MAE_short: Short term MAE results.
        MAE_long: Long term MAE results.
        bias_short: Short term bias results.
        bias_long: Long term bias results.
        rel_bias_short: Short term relative bias results.
        long_bias_short: Long term relative bias results.
    """
    
    # Import patterns
    pattern_ds_path = f'{path_to_ERF_outputs}Patterns/pattern_{train_id}_ds.nc4'
    pattern_ds = xr.open_dataset(pattern_ds_path)
    
    # Arrays for storing error statistics
    RMSE_short, RMSE_long = [], []
    MAE_short, MAE_long = [], []
    bias_short, bias_long = [], []
    rel_bias_short, rel_bias_long = [], []
    
    for scale_id in scale_all:
        print(f'\tLoading {scale_id} experiment for pattern scaling...')
        
        tas_ds_path = f'{path_to_ERF_outputs}tas/tas_{scale_id}_ds.nc4'
        tas_ds = xr.open_dataset(tas_ds_path) 

        # Perform pattern scaling
        if 'model' in pattern_ds:
            pattern = pattern_ds.mean(dim = ['model'])
        else:
            pattern = pattern_ds

        tas_glob_ds = tas_ds.weighted(A).mean(dim = ['lat','lon']).mean(dim = ['model'])
        pattern_manip = pattern.assign_coords({'s':tas_glob_ds.s}).stack(allpoints=['lat','lon'])
        scaled_vals = pattern_manip.pattern.values * tas_glob_ds.tas.values[:, np.newaxis]

        scaled_pattern = xr.Dataset(coords={'lon': ('lon', pattern.lon.values),
                                      'lat': ('lat', pattern.lat.values),
                                      's':tas_glob_ds.s.values})

        scaled_pattern = scaled_pattern.stack(allpoints=['lat','lon'])
        scaled_pattern['tas'] = (('s','allpoints'),scaled_vals)
        scaled_pattern = scaled_pattern.unstack('allpoints')
    
        if save_result:
            scaled_pattern.to_netcdf(f'{output_path}Spatial Results/res_pattern_spatial_{train}_{scale}_ds.nc4') 

        # Select short and long term time periods based on experiment
        if 'ssp' in scale_id: # SSP experiments
            start_yr1, plot_yr1, end_yr1 = 2040, 2050, 2060
            start_yr2, plot_yr2, end_yr2 = 2080, 2090, 2100
        elif 'hist' in scale_id: # Historical experiment
            start_yr1, plot_yr1, end_yr1 = 1900, 1920, 1940
            start_yr2, plot_yr2, end_yr2 = 1985, 2000, 2015
        else: # 1pctCO2 experiment
            start_yr1, plot_yr1, end_yr1 = 1940, 1950, 1960
            start_yr2, plot_yr2, end_yr2 = 1980, 1990, 2000
        
        # Make plots for analysis, comment out lines depending on desired plots
        if plot:
            plot_pattern(scaled_pattern, train_id, scale_id, tas_ds, plot_yr1, 10, save_fig = save_fig)
            plot_pattern(scaled_pattern, train_id, scale_id, tas_ds, plot_yr2, 10, save_fig = save_fig)
            
        # Calculate and record error values
        MSE1, RMSE1, MAE1, bias1, rel_bias1 = calc_error_metrics(tas_ds, scaled_pattern, start_yr1, end_yr1, mean_GF = False, pattern = True)
        RMSE_short.append(round(RMSE1,4))
        MAE_short.append(round(float(MAE1),4))
        bias_short.append(round(float(bias1),4))
        rel_bias_short.append(round(float(rel_bias1),4))

        MSE2, RMSE2, MAE2, bias2, rel_bias2 = calc_error_metrics(tas_ds, scaled_pattern, start_yr2, end_yr2, mean_GF = False, pattern = True)
        RMSE_long.append(round(RMSE2,4))
        MAE_long.append(round(float(MAE2),4))
        bias_long.append(round(float(bias2),4))
        rel_bias_long.append(round(float(rel_bias2),4))
        
    # Print results
    if verbose:
        print('\nResults are shown in the following order:')
        print(scale_all)

        print('\nMid-Century Stats:')
        print(f'RMSE: {RMSE_short}')
        print(f'MAE: {MAE_short}')
        print(f'Bias: {bias_short}')
        print(f'Relative Bias: {rel_bias_short}')

        print('\nEnd-of-Century Stats:')
        print(f'RMSE: {RMSE_long}')
        print(f'MAE: {MAE_long}')
        print(f'Bias: {bias_long}')
        print(f'Relative Bias: {rel_bias_long}')
    
    return RMSE_short, RMSE_long, MAE_short, MAE_long, bias_short, bias_long, rel_bias_short, rel_bias_long

######################## Plotting ###########################

def plot_ERF_profile(ERF_ds, exp_id, save_fig = False):
    """
    Plots the ERF profile for a given experiment.
    
    Args:
        ERF_ds: Dataset of ERF data.
        conv_id: Experiment ID the ERF profile is taken from.
        save_fig: Whether or not to save the figure.
        
    Returns:
        None.
    """
    
    fig, ax = plt.subplots(figsize = [10,6])
    for m in ERF_ds.model:
        ax.plot(ERF_ds['year'],ERF_ds.sel(model = m)['ERF'], alpha = .8, label = f'{m.values}')
    ax.plot(ERF_ds['year'],ERF_ds.mean(dim = 'model')['ERF'], color = 'k', label = f'Ensemble Mean')
    
    ax.legend(fontsize = 14)
    ax.set_xlabel('Year', fontsize = 20)
    ax.set_ylabel('Wm$^{-2}$', fontsize = 20)
    ax.set_title(f'{conv_id_cap[exp_id]} $F$ Profile', fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    if save_fig:
        plt.savefig(f'{path_to_figures}{conv_id}/ERF_profile_{conv_id}.pdf', bbox_inches = 'tight', dpi = 500)
        
    return

def plot_conv_meanGF(train_id, conv_id, conv_mean_ds, tas_ds, sens = False, conv_mean_p10_ds = None, conv_mean_m10_ds = None, save_fig = False):
    """
    Plots the emulated global mean temperature for a given experiment.
    
    Args:
        train_id: Experiment ID of the training dataset.
        conv_id: Experiment ID of the test dataset being convolved with.
        conv_mean_ds: Dataset containing the results of convolution.
        tas_ds: Dataset containing ground truth near surface air temperature for comparison.
        sens: Whether or not the data comes from a sensitivity analysis.
        conv_mean_p10_ds: Dataset containing the results of convolution with a +10% \lambda.
        conv_mean_m10_ds: Dataset containing the results of convolution with a -10% \lambda.
        save_fig: Whether or not to save the figure.
        
    Returns:
        None.
    """
    
    fig, ax = plt.subplots(figsize = [8,6])
    
    ax.plot(conv_mean_ds['s'] + 1850, conv_mean_ds.sel(train_id = train_id), label = 'Emulated tas',color=brewer2_light(2),linewidth = 3)
    
    if sens:
        ax.fill_between(conv_mean_ds['s'] + 1850, conv_mean_m10_ds.sel(train_id = train_id), conv_mean_p10_ds.sel(train_id = train_id),color=brewer2_light(2),alpha=0.5)
        
    ax.plot(np.arange(1850,1850 + len(tas_ds['s'])), tas_ds.mean(dim = 'model').weighted(A).mean(dim = ['lat','lon'])['tas'], 
         label = 'CMIP6 Ensemble Mean tas', linewidth = 3, linestyle = '-.',color=brewer2_light(1))
    
    ax.legend(fontsize = 18)
    ax.set_xlabel('Year', fontsize = 20)
    ax.set_ylabel('$\overline{\Delta T}\,(t)$ [$\degree$C]', fontsize = 20)
    ax.set_title(f'Global Mean Temperature Emulation\nPredictor: {train_id}, Target: {conv_id_cap[conv_id]}',fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.grid(True)
    fig.tight_layout()
    
    if save_fig:
        if sens:
            plt.savefig(f'{path_to_figures}conv_global_sens_{train_id}_{conv_id}.pdf', bbox_inches = 'tight', dpi = 500)
        else:
            plt.savefig(f'{path_to_figures}conv_global_{train_id}_{conv_id}.pdf', bbox_inches = 'tight', dpi = 500)
        
    return

def plot_dif_map_meanGF(conv_ds, tas_ds, plot_yr, yr_dif, conv_id, train_id, dif = True, save_fig = False):  
    """
    Plots either a map of the difference between the CMIP6 ground truth data and the emulator
    or a map of the raw predicted temperature change.
    
    Args:
        conv_ds: Dataset containing convolved temperature response.
        tas_ds: Dataset containing ground truth temperature response.
        plot_yr: Year for plotting (>= 1850).
        yr_dif: Number of years to average the plot over.
        conv_id: ID of experiment to convolve with.
        train_id: ID of experiment Green's function was trained on.
        dif: Whether or not to plot a difference map.
        save_fig: Whether or not to save the figure.
        
    Returns:
        None.
    """
    
    plot_yr = plot_yr - 1850
    cmap = mpl.cm.RdBu_r
    fig, ax= plt.subplots(figsize = [10,6], subplot_kw = {'projection':ccrs.Robinson()}, constrained_layout = True)

    tas_ds = tas_ds.mean(dim = 'model')
    
    # Contours of difference
    if dif:
        extremes = (-2, 2)
        norm = plt.Normalize(*extremes)
        data = (conv_ds -  tas_ds['tas']).mean(dim = 'train_id').sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's')
        im = ax.pcolormesh(conv_ds.lon, conv_ds.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm = norm)
    
    else:
        extremes = (0, 9)
        norm = plt.Normalize(*extremes)
        data = tas_ds['tas'].sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's')
        im = ax.pcolormesh(tas_ds.lon, tas_ds.lat, data, transform=ccrs.PlateCarree(), cmap=mpl.cm.Reds, norm = norm)
    
    ax.coastlines()
    cb = fig.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.7, extend = 'both')
    cb.set_label('$\Delta T$ [$\degree$C]', fontsize = 20)
    cb.ax.tick_params(labelsize=16)
    
    if dif:
        ax.set_title(f'(Emulator - CMIP6) in {plot_yr + 1850} ($\pm {yr_dif}$ years)\nPredictor: {train_id}, Target: {conv_id_cap[conv_id]}', fontsize = 20)
    else:
        ax.set_title(f'Temperature Change Relative to 1850 in {plot_yr + 1850} ($\pm {yr_dif}$ years)\nCMIP6 Scenario: {conv_id_cap[conv_id]}', fontsize = 20)
    
    if save_fig:
        if dif:
            plt.savefig(f'{path_to_figures}conv_spatial_dif_{train_id}_{conv_id}_{plot_yr}.pdf', bbox_inches = 'tight', dpi = 500)
        else:
            plt.savefig(f'{path_to_figures}cmip_spatial_raw_{train_id}_{conv_id}_{plot_yr}.pdf', bbox_inches = 'tight', dpi = 500)
        
    return

def plot_pattern(pattern_ds, train_id, test_id = None, tas_ds = None, plot_yr = None, yr_dif = None, save_fig = False):
    """
    Plots either a map of the warming pattern for a given experiment or a map of the difference between
    the pattern scaled temperature response and the CMIP6 ground truth.
    
    Args:
        pattern_ds: Dataset containing warming pattern.
        train_id: ID of experiment pattern was trained on.
        test_id: Experiment being pattern scaled.
        tas_ds: CMIP6 ground truth data.
        plot_yr: Year for plotting (>= 1850).
        yr_dif: Number of years to average the plot over.
        save_fig: Whether or not to save the figure.
        
    Returns:
        None.
    """
    
    cmap = mpl.cm.RdBu_r
    fig, ax= plt.subplots(figsize = [10,6], subplot_kw = {'projection':ccrs.Robinson()}, constrained_layout = True)
            
    if tas_ds:
        plot_yr = plot_yr - 1850
        extremes = (-2, 2)
        norm = plt.Normalize(*extremes)
        data = (tas_ds.mean(dim = 'model') - pattern_ds).sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's')['tas']
        im = ax.pcolormesh(pattern_ds.lon, pattern_ds.lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm = norm)
        
    else:
        extremes = (-2, 2)
        norm = plt.Normalize(*extremes)
        im = ax.pcolormesh(pattern_ds.lon, pattern_ds.lat, pattern_ds['pattern'], transform=ccrs.PlateCarree(), cmap=cmap, norm = norm)     

    ax.coastlines()
    cb = fig.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.7, extend = 'both')
    cb.set_label('$\Delta T$ [$\degree$C]', fontsize = 20)
    cb.ax.tick_params(labelsize=16)
    
    if tas_ds:
        ax.set_title(f'(Emulator - CMIP6) in {plot_yr + 1850} ($\pm {yr_dif}$ years)\nPredictor: {train_id}, Target: {test_id}', fontsize = 20)
        #ax.set_title(f'{train_id}_{test_id}: Difference at {1850 + plot_yr} ($\pm {yr_dif}$) years', fontsize = 14)
    else:
        ax.set_title(f'Warming Pattern, Trained on {train_id}', fontsize = 20)
        #ax.set_title(f'{train_id}: Pattern', fontsize = 14)
    
    if save_fig:
        if tas_ds:
            plt.savefig(f'{path_to_figures}patt_spatial_dif_{train_id}_{conv_id}_{plot_yr}.pdf', bbox_inches = 'tight', dpi = 500)
        else:
            plt.savefig(f'{path_to_figures}patt_spatial_raw_{train_id}_{conv_id}_{plot_yr}.pdf', bbox_inches = 'tight', dpi = 500)
        
    return 

######################## Error Metrics ###########################

def calc_error_metrics(truth, emulator, start_year, end_year, mean_GF = True, pattern = False):
    """
    Function to calculate RMSE, MAE, and bias given a ground truth dataset and emulator dataset.
    
    Args:
        truth: Ground truth dataset.
        emulator: Emulator dataset.
        start_year: Year in which to begin comparison.
        end_year: Year in which to end comparison.
        mean_GF: Whether or not to average the emulator results over all models.
        pattern: Whether or not the comparison is made relative to a pattern scaling emulator.
        
    Returns:
        MSE: Mean square error.
        RMSE: Root mean square error.
        MAE: Mean absolute error.
        bias: Bias.
        rel_bias: Relative bias (%).
    """
    
    # Datasets are organized starting at zero rather than 1850
    slice_start = start_year - 1850
    slice_end = end_year - 1850
    
    # Average CMIP ground truth over all models
    truth = truth.mean(dim = 'model')
    
    # Get emulator data into correct format
    if mean_GF and pattern == False:
        emulator = emulator.mean(dim = 'model').mean(dim = 'train_id').sel(s = slice(min(truth.s),max(truth.s)))
    elif pattern == False:
        emulator = emulator.mean(dim = 'train_id').sel(s = slice(min(truth.s),max(truth.s)))
    
    # Ensure time dimensions line up for both datasets
    truth = truth.sel(s = slice(slice_start,slice_end))
    emulator = emulator.sel(s = slice(slice_start,slice_end))
    
    # Calculate RMSE
    if pattern:
        MSE = np.square(np.subtract(truth['tas'],emulator['tas'])).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        MSE = np.square(np.subtract(truth['tas'],emulator)).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    RMSE = math.sqrt(MSE)
    
    # Calculate MAE
    if mean_GF and pattern == False:
        MAE = np.abs(np.subtract(truth['tas'],emulator)).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s']).mean(dim = ['model'])
    elif pattern == False:
        MAE = np.abs(np.subtract(truth['tas'],emulator)).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        MAE = np.abs(np.subtract(truth['tas'],emulator['tas'])).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    
    # Calculate bias
    if pattern:
        bias = np.subtract(emulator['tas'],truth['tas']).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        bias = np.subtract(emulator,truth['tas']).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
        
    # Calculate relative bias
    if pattern:
        rel_bias = 100*np.divide(np.subtract(emulator['tas'],truth['tas']),truth['tas']).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        rel_bias = 100*np.divide(np.subtract(emulator,truth['tas']),truth['tas']).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])

    return MSE.values, RMSE, MAE.values, bias.values, rel_bias.values

def calc_area_error(truth_path, emulator_path, start_year, end_year):
    """
    Function to calculate relative biases by the percent coverage of the Earth's surface area.
    This function is NOT optimized and is very slow, could be improved significantly.
    
    Args:
        truth_path: Path to ground truth dataset.
        emulator_path: Path to emulator dataset.
        start_year: Year in which to begin comparison.
        end_year: Year in which to end comparison.
        
    Returns:
        area_pct: List corresponding to the pct area coverage by each level of relative bias.
        bins: Edges of each bin for the histogram calculation.
        mn: Mean of relative biases.
        std: Standard deviation of relative biases.
    """

    # Open ground truth dataset
    truth = xr.open_dataset(truth_path) 

    # Different datastructures for pattern scaling vs. GFs
    if 'patt' in emulator_path:
        emulator = xr.open_dataset(emulator_path)
    else:
        emulator = xr.open_dataset(emulator_path)['__xarray_dataarray_variable__']
        emulator.name = 'tas'
    
    # Shift time
    slice_start = start_year - 1850
    slice_end = end_year - 1850
    
    # Average ground truth by model and ensure times line up
    truth = truth.mean(dim = 'model')
    truth = truth.sel(s = slice(slice_start,slice_end))    
    
    # Correct emulator data, dataset was setup for multiple training runs
    if 'conv' in emulator_path:
        emulator = emulator.mean(dim = 'train_id')
    
    # Ensure emulator times line up
    emulator = emulator.sel(s = slice(slice_start,slice_end))
    
    # For each grid point, calculate % error from truth
    pct_error_spatial = np.divide(np.subtract(emulator,truth['tas']),truth['tas']).mean(dim = 's')

    # We have a few grid cells bugged in ssp126 when performing this operation, they vary between the short and long term
    # Convert them to NaN 
    if 'ssp126' in truth_path and '1pctCO2' in emulator_path and start_year == 2040:
        pct_error_spatial.loc[{'lat': -62.5, 'lon': 153}] = np.nan
    elif 'ssp126' in truth_path and '1pctCO2' in emulator_path and start_year == 2080:
        pct_error_spatial = pct_error_spatial.where(pct_error_spatial > -10, np.nan)
        pct_error_spatial = pct_error_spatial.where(pct_error_spatial < 10, np.nan)
   
    # Calculate mean and standard deviation of relative biases
    mn = pct_error_spatial.weighted(A).mean(dim = ['lat','lon'])
    std = pct_error_spatial.weighted(A).std(dim = ['lat','lon'])
    
    # Stack points for later calculation
    pct_error_stacked = pct_error_spatial.stack(allpoints=['lat','lon'])
    
    # Create dataset of total area
    total_area = xr.DataArray(np.ones((len(truth.lon.values),len(truth.lat.values))),
                              coords={'lon': ('lon', truth.lon.values),
                              'lat': ('lat', truth.lat.values)})
    total_area.to_dataset(name = 'area')

    # A contains surface areas for all grid cells, allows us to compute surface area at a given 
    # level of error
    area_stack = A.stack(allpoints=['lat','lon'])

    # Create histogram of error by area on the planet 
    area_pct = []
    n_bins = 100
    if 'patt' in emulator_path:
        counts, bins = np.histogram(pct_error_stacked.tas.values, range = (-0.25,0.25), bins = n_bins)
    else:
        counts, bins = np.histogram(pct_error_stacked.values, range = (-0.25,0.25), bins = n_bins)
    seen = set()
    for i in range(n_bins):
        bin_l = bins[i]
        bin_r = bins[i+1]
        area_count = 0
        for j in range(len(pct_error_stacked['allpoints'].values)):
            if j in seen:
                continue

            if 'patt' in emulator_path:
                test_val = pct_error_stacked.tas.values[j]
            else:
                test_val = pct_error_stacked.values[j]

            if bin_l <= test_val <= bin_r:
                area_count += area_stack[j]
                seen.add(j)

        area_pct.append(area_count)
        
    # Scale each bin by the total surface area of the Earth
    surf_earth = sum(sum(A))
    for i in range(n_bins):
        if type(area_pct[i]) is int:
            area_pct[i] = area_pct[i]/surf_earth
        else:
            area_pct[i] = area_pct[i].values/surf_earth
    
    return area_pct, bins, mn, std

########################### General Helper Functions ###################################

##### Local weighted regression, credit:
# https://www.geeksforgeeks.org/locally-weighted-linear-regression-using-python/
def local_weighted_regression(x0, X, Y, tau):
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
     
    # fit model: normal equations with kernel
    xw = X.T * weights_calculate(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    # "@" is used to
    # predict value
    return x0 @ theta

def weights_calculate(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau **2) ))

#####

def concat_multirun(raw_data, name):
    """
    Concatenates data from multiple runs.
    
    Args:
        raw_data: Dictionary to be concatenated.
        name: Dimension along which to concatenate.
        
    Returns:
        Concatenated dataset.
    """
    return xr.concat([raw_data[m] for m in raw_data.keys()], pd.Index([m for m in raw_data.keys()], name=name), coords='minimal')

def ds_to_dict(ds):
    
    ds_dict = {}
    for m in ds['model']:
        ds_dict[str(m.values)] = ds.sel(model = m)
    
    return ds_dict

def monthly_to_annual(ds):
    """
    Converts monthly CMIP data to annual data, weighting by month length.
    
    Args:
        ds: Dataset to convert.
        
    Returns:
        ds: Converted dataset.
    """
    times = ds.time.get_index('time')
    weights = times.shift(-1, 'MS') - times.shift(1, 'MS')
    weights = xr.DataArray(weights, [('time', ds['time'].values)]).astype('float')
    ds = (ds * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
    
    return ds

### Helper functions to convert numpy arrays to xarray datasets
# Credit: https://github.com/lfreese/CO2_greens
def np_to_xr(C, G, E):
    E_len = len(E)
    G_len = len(G.s)
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0, C.shape[0])), #np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return(C)
def np_to_xr_mean(C, G, E):
    E_len = len(E)
    G_len = len(G.s)
    C = xr.DataArray(
    data = C,
    dims = ['s'],
    coords = dict(
        s = (['s'], np.arange(0, C.shape[0])), #np.arange(0,(E_len+G_len))),
            )
        )
    return(C)

### Define output grid size
# Credit: https://github.com/lfreese/CO2_greens
ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
        "lon": (["lon"], np.arange(0, 360, 1)),
        "lat_b": (["lat_b"], np.arange(-90.,91.,1.0)),
        "lon_b":(["lon_b"], np.arange(.5, 361.5, 1.0))
    }
)

#### function to find area of a grid cell from lat/lon ####
# Credit: https://github.com/lfreese/CO2_greens
def find_area(ds, R = 6378.1):
    """
    ds is the dataset, i is the number of longitudes to assess, j is the number of latitudes, and R is the radius of the earth in km. 
    Must have the ds['lat'] in descending order (90...-90)
    Returns Area of Grid cell in km
    """
    
    circumference = (2*np.pi)*R
    deg_to_m = (circumference/360) 
    dy = (ds['lat_b'].roll({'lat_b':-1}, roll_coords = False) - ds['lat_b'])[:-1]*deg_to_m

    dx1 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b']))
    
    dx2 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b'].roll({'lat_b':-1}, roll_coords = False)[:-1]))
    
    A = .5*(dx1+dx2)*dy
    
    #### assign new lat and lon coords based on the center of the grid box instead of edges ####
    A = A.assign_coords(lon_b = ds.lon.values,
                    lat_b = ds.lat.values)
    A = A.rename({'lon_b':'lon','lat_b':'lat'})

    A = A.transpose()
    
    return(A)

A = find_area(ds_out)

######################## Dataset Dictionaries ###########################

# Climate feedback parameters, credit: Nijsse et al. (2020),
# https://esd.copernicus.org/articles/11/737/2020/
lam_dict = {'ACCESS-CM2':0.67,
            'ACCESS-ESM1-5':0.68, 
            'CAMS-CSM1-0':1.71, 
            'CanESM5':0.64,
            'INM-CM4-8':1.42,
            'INM-CM5-0':1.49,
            'MIROC6':1.47,
            'MRI-ESM2-0':1.07,
            'NorESM2-LM':1.13}

# Set of models for diagnosis
model_set = set(['ACCESS-CM2',
                 'ACCESS-ESM1-5',
                 'CAMS-CSM1-0',
                 'CanESM5',
                 'INM-CM4-8',
                 'INM-CM5-0',
                 'MIROC6',
                 'MRI-ESM2-0',
                 'NorESM2-LM'])

# Sets of models for debugging
model_test_set = set(['MIROC6','ACCESS-ESM1-5','CAMS-CSM1-0'])
model_test_single_set = set(['MIROC6'])

conv_id_cap = {'1pctCO2':'1pctCO2',
               'ssp126':'SSP126',
               'ssp245':'SSP245',
               'ssp370':'SSP370',
               'ssp585':'SSP585'}

######################## Plot Colormap ###########################
# Credit: https://colorbrewer2.org/#type=qualitative&scheme=Set2&n=8
brewer2_light_rgb = np.divide([(102, 194, 165),
                               (252, 141,  98),
                               (141, 160, 203),
                               (231, 138, 195),
                               (166, 216,  84),
                               (255, 217,  47),
                               (229, 196, 148),
                               (179, 179, 179)],255)
brewer2_light = mcolors.ListedColormap(brewer2_light_rgb)
