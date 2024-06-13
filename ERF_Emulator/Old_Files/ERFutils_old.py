import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cftime
import dask
import xarrayutils
import cartopy.crs as ccrs
from xmip.preprocessing import combined_preprocessing
from xmip.preprocessing import replace_x_y_nominal_lat_lon
from xmip.drift_removal import replace_time
from xmip.postprocessing import concat_experiments
import xmip.drift_removal as xm_dr
import xmip as xm
import xesmf as xe
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import cf_xarray as cfxr
import scipy.signal as signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve_triangular
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import math

import os
import seaborn as sns
import matplotlib as mpl
import cmocean
import cmocean.cm as cmo
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# Define matplotlib style and set everything to LaTeX font
matplotlib.rcdefaults()
plt.style.use('seaborn-v0_8-colorblind')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

########################### Path Definitions ###################################

path_to_cmip6_data_local = '../../EnROADS_CO2_Greens/cmip6_data/'
path_to_cmip6_data = path_to_cmip6_data_local
path_to_cmip6_data_clean = '../../EnROADS_CO2_Greens/cmip6_data_clean/'
path_to_piControl = f'{path_to_cmip6_data}piControl/'
path_to_ERF_outputs = '../ERF_Outputs/'
path_to_figures = f'{path_to_ERF_outputs}JAMES_Figures/'

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

def load_ERF_set(model_set, data_id, print_var = False):
    """
    Loads ERF for a given dataset.
    
    Args:
        model_set: Set of model names within the given dataset.
        data_id: ID of the dataset, e.g. 1pctCO2.
        print_var: Boolean to print variable path, used for debugging.
        ssp: Boolean to denote if dataset is an SSP scenario, used for syncing times.
        lambda_error: List of percentages indicating if we allow for error in climate sensitivity.
        
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
            if print_var:
                print(f'{path_to_cmip6_data_clean}{data_id}/{var}_Amon_{m}_{data_id}_r1i1p1f1.nc4')
                print(f'{path_to_cmip6_data_clean}piControl/{var}_Amon_{m}_piControl_r1i1p1f1.nc4')
                
            # Load variable and remove climatology
            EB_dict[m][var] = remove_climatology(data_id, m, var)

        # Calculate energy balance
        EB_dict[m]['RF'] = calc_energy_balance(EB_dict[m]['rlut']['rlut'], EB_dict[m]['rsdt']['rsdt'], EB_dict[m]['rsut']['rsut'])    
        ERF[m] = (EB_dict[m]['RF'].RF + lam[m]*EB_dict[m]['tas'].tas).to_dataset(name = 'ERF')
    
    return ERF

########################### CMIP 6 DATA AND REGRIDDING ###################################


##### Local weighted regression, taken from:
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

def mf_check(m):
    ds = xr.open_mfdataset(f'{path_to_cmip6_data}1pctCO2/tas_Amon_{m}_1pctCO2_r1i1p1f1**', use_cftime=True)
    return ds

def mf_to_sf(data_id, m, var):
    print(f'Checking {data_id}, {m}, {var}...')
    path_clean = f'{path_to_cmip6_data_clean}{data_id}/{var}_Amon_{m}_{data_id}_r1i1p1f1.nc4'
    
    if os.path.isfile(path_clean):
        return
    
    path_mf = f'{path_to_cmip6_data}{data_id}/{var}_Amon_{m}_{data_id}_r1i1p1f1**'
    ds = xr.open_mfdataset(path_mf, use_cftime=True)
    ds.chunk(1000000)
    ds.to_netcdf(path_clean)
    
    return

def calc_ensemble_variability(m_set):
    piControl = {}
    for m in m_set:
        print(m)
        # Load and convert piControl data to yearly
        piControl[m] = xr.open_mfdataset(f'{path_to_cmip6_data}piControl/tas_Amon_{m}_piControl_r1i1p1f1.nc4', use_cftime=True, chunks = 1000000, parallel=True)
        piControl[m] = monthly_to_annual(piControl[m])
        piControl[m] = piControl[m].weighted(A).mean(dim = ['lat','lon'])
        
    piControl = concat_multirun(piControl,'model').mean(dim = 'model')
    
    # Average piControl over the entire simulation period
    variability = piControl.std(dim = ['year'])
    return variability.as_numpy()


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
    piControl_ds = xr.open_mfdataset(f'{path_to_cmip6_data_clean}piControl/{var}_Amon_{m}_piControl_r1i1p1f1.nc4', use_cftime=True, chunks = 1000000)
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
    ds = xr.open_dataset(f'{path_to_cmip6_data_clean}{data_id}/{var}_Amon_{m}_{data_id}_r1i1p1f1.nc4', use_cftime=True, chunks = 1000000)
    ds = monthly_to_annual(ds).as_numpy()
    
    # Calculate climatology
    climatology = calc_climatology(m, var)
    
    # Remove climatology from dataset
    ds_rem = (ds[var] - climatology[var]).to_dataset(name = var)
    return ds_rem

def sync_time(ds_train, ds_control, ssp=False):
    if ssp:
        ds_control['year'] = ds_control['year'] - ds_control['year'].values[0] + ds_train['year'].values[0] - 164
        ds_control = ds_control.sel(year = slice(ds_train['year'].min(), ds_train['year'].max()))
    
    if 'year' in ds_control:
        ds_control['year'] = ds_control['year'] - ds_control['year'].values[0] + ds_train['year'].values[0]
        ds_control = ds_control.sel(year = slice(ds_control['year'].min(), ds_train['year'].max()))
    else:
        ds_control['time'] = ds_control['time'] - ds_control['time'].values[0] + ds_train['time'].values[0]
        ds_control = ds_control.sel(time = slice(ds_control['time'].min(), ds_train['time'].max()))
    
    return ds_train, ds_control

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

def create_multimodel_GF(RF, train_id, model_control_dict, model_train_dict, savgol = False, save_GF = False):
    """
    Creates Green's Functions from multiple models given a set of training data.
    
    Args:
        RF: Radiative forcing (or variable to create GF from).
        train_id: ID of the training run, e.g. 1pct-brch-1000PgC.
        model_control_dict: Dictionary of models with piControl runs.
        model_train_dict: Diction of models in training dataset.
        save_GF: Whether or not to save out the resulting GF.
        
    Returns:
        ds_control: Dataset with piControl data from all models.
        ds_train: Dataset with training data from all models.
        G_ds: Dataset with GFs for each model with the given training data.
    """
    
    ds_control, ds_train, G = {}, {}, {}
    for m in model_control_dict.keys():
        if m not in model_train_dict: raise Exception(f'Control model, {m}, not in training dictionary.')
        print(f'Diagnosing GF for {m}...')
        ds_control[m], ds_train[m], G[m] = import_regrid_calc_deconv(f'{path_to_piControl}tas_Amon_{model_control_dict[m]}',
                                                                                         f'{path_to_cmip6_data}{train_id}/tas_Amon_{model_train_dict[m]}',
                                                                                         RF[train_id][m],
                                                                                         ds_out, m, savgol)

    G_ds = concat_multirun(G, 'model')
    
    if save_GF:
        G_ds.to_netcdf(f'{path_to_ERF_outputs}/G_{train_id}_ERF_ds.nc4')
    
    return ds_control, ds_train, G_ds

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
    ERF_path = f'{path_to_ERF_outputs}ERF/ERF_{train}_all_ds.nc4'
    ERF_ds = xr.open_dataset(ERF_path)
    ERF[train] = ds_to_dict(ERF_ds)
    ERF_all = concat_multirun(ERF[train],'model').mean(dim = 'model')

    # Load tas data
    tas_path = f'{path_to_ERF_outputs}tas/tas_CMIP_{train}_all_ds.nc4'
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
        fig, ax = plt.subplots(figsize = [10,6])
        ax.plot(G_ds.weighted(A).mean(dim = ['lat','lon']).values[0:31], linewidth=3)

        ax.set_title(f'Global Average Green\'s Function: 1pctCO2 Experiment',fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax.set_xlabel('Years Since ERF Impulse',fontsize=20)
        ax.set_ylabel('Impulse Response [$^\circ$C/(Wm$^{-2}$)]',fontsize=20)
        plt.grid(True)

        fig.tight_layout()
        
        if save_fig:
            plt.savefig(f'{path_to_figures}global_GF_{train}.pdf', bbox_inches = 'tight', dpi = 350)

    # Save resultant Green's function
    if save_data:
        G_ds.to_netcdf(f'{path_to_ERF_outputs}G_{train}_ds.nc4')
    
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
    tas_path = f'{path_to_ERF_outputs}tas/tas_CMIP_{train}_all_ds.nc4'
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
            plt.savefig(f'{path_to_figures}global_pattern_{train}.pdf', bbox_inches = 'tight', dpi = 350)

    # Save resultant pattern
    if save_data:
        pattern.to_netcdf(f'{path_to_ERF_outputs}pattern_{train}_all_ds.nc4')
    
    return pattern_ds

def import_regrid_calc_deconv(input_signal, train_id, ds_out, m, savgol = False, regrid = True):
    """
    Imports the control run and pulse run for a CMIP6 model run, combines them on the date the pulse starts. 
    Regrids it to the chosen grid size and calculates the Green's Function.
    
    Args:
        input_signal: Variable to create GF from.
        train_id: ID of the training run, e.g. 1pctCO2.
        ds_out: Size of the grid you want, must have lat, lon, lat_b, lon_b.
        m: Model name. naming convention for the models is MODELID.
        regrid: True or False (auto to True).
        
    Returns:
        ds_control: Control data.
        ds_train: Training data.
        G: Green's Function for a given experiment.
    """
    
    control_path = f'{path_to_cmip6_data_clean}piControl/tas_Amon_{m}_piControl_r1i1p1f1.nc4'
    train_path = f'{path_to_cmip6_data_clean}{train_id}/tas_Amon_{m}_{train_id}_r1i1p1f1.nc4'
    
    ds_control, ds_train = import_combine_train_control(control_path, train_path, m)
    if regrid == True:
        ds_control = regrid_cmip(ds_control, ds_out)
        ds_train = regrid_cmip(ds_train, ds_out)
        #input_signal = regrid_cmip(input_signal, ds_out)
    
    G = calc_greens_deconv(ds_control, ds_train, input_signal, m, savgol)
    
    return ds_control, ds_train, G

def regrid_corners(ds):
    lat_corners = cfxr.bounds_to_vertices(ds.isel(time = 0)['lat_bnds'], "bnds", order=None)
    lon_corners = cfxr.bounds_to_vertices(ds.isel(time = 0)['lon_bnds'], "bnds", order=None)
    ds = ds.assign(lon_b=lon_corners, lat_b=lat_corners)
                      
    return ds
                      
def import_combine_train_control(control_path, train_path, m):
    """
    Import the pulse run and control run for each model. 
    
    Args:
        control_path: Path to the control run.
        train_path: Path to training run.
        m: Model name. naming convention for the models is MODELID.
        
    Returns:
        ds_control: Control dataset.
        ds_train: Training dataset.
    """
    
    # Open the control and pulse runs
    ds_control = xr.open_dataset(control_path, use_cftime=True, chunks = 1000000)
    ds_train = xr.open_dataset(train_path, use_cftime=True, chunks = 1000000)
    
    # Find and reassign lat and long corners/bounds
    ds_control = regrid_corners(ds_control)
    ds_train = regrid_corners(ds_train)

    # Check that we are bringing in a control and training run from the same model id
    if ds_control.attrs['parent_source_id'] != ds_train.attrs['parent_source_id']: 
        print('WARNING: Control and Training runs are not from the same parent source!')
    
    # Select only the times that match up with the experiment
    ds_train, ds_control = sync_time(ds_train, ds_control)

    return ds_control, ds_train

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

def calc_pattern(train_id, m):
    """
    Calculate the climate pattern given a scenario. 
    
    Args:
        train_id: Training run ID.
        m: Model name.
        
    Returns:
        pattern
    """
    
    # Get temperature response as difference between training and control runs
    temp_response = RFutils.remove_climatology(train_id, m, 'tas')
    temp_response = RFutils.regrid_cmip(temp_response,ds_out)
    global_temp = temp_response.weighted(A).mean(dim = ['lat','lon']).tas.values

    # Have to create the patterns locally, stack data array
    stacked_response = temp_response.stack(allpoints=['lat','lon'])
    N_latlong = len(stacked_response['allpoints'].values)

    # Convert to np arrays, xarray indexing is too slow
    pattern_stacked = np.zeros((1,N_latlong))
    stacked_response_np = stacked_response.tas.values

    # Solve for spatially resolved pattern
    for i in range(N_latlong):
        stacked_response_local = stacked_response_np[:,i]
        reg = LinearRegression().fit(global_temp.reshape(-1,1), stacked_response_local.reshape(-1,1))
        pattern_stacked[0,i] = reg.coef_
    
    pattern = xr.Dataset(coords={'lon': ('lon', temp_response.lon.values),
                        'lat': ('lat', temp_response.lat.values)})
    pattern = pattern.stack(allpoints=['lat','lon'])
    pattern['pattern'] = ('allpoints',pattern_stacked[0])
    pattern = pattern.unstack('allpoints')
        
    return pattern

def plot_mean_Greens(G_ds, train_id, overlay = True, save_fig = False):
    
    if overlay:
        fig, ax = plt.subplots(figsize = [10,6])
        mean_models = {}
        
    for m in G_ds.model:
        #if str(m.model.values) == 'MIROC' or str(m.model.values) == 'CAMS':
        #    continue
        if overlay:
            mean_global = G_ds.sel(model=m).weighted(A).mean(dim = ['lat','lon'])
            mean_models[str(m.model.values)] = mean_global
            #ax.plot(mean_global['__xarray_dataarray_variable__'],label=str(m.model.values))
            ax.plot(mean_global,label=str(m.model.values))
            ax.set_title(f'Global Mean GFs for {train_id}')
            ax.set_xlabel('Years Since RF Pulse')
            ax.set_ylabel('GF (K/(W/m^2))')
            ax.legend()
            
        else:
            fig, ax = plt.subplots()
            ax.plot(G_ds.sel(model=m).weighted(A).mean(dim = ['lat','lon']))
            ax.set_title(f'Global Mean GF for {m}')
            ax.set_xlabel('Years Since RF Pulse')
            ax.set_ylabel('GF (K/(W/m^2))')
            
            if save_fig:
                plt.savefig(f'{path_to_figures}{train_id}/{m}_GF.png', bbox_inches = 'tight', dpi = 350)
              
    if overlay:
        mean_model_ds = concat_multirun(mean_models,'model')
        ax.plot(mean_model_ds.mean(dim = ['model']),label='Ensemble Mean',color='black')
        ax.set_title(f'Global Mean GFs for {train_id}')
        ax.set_xlabel('Years Since RF Pulse')
        ax.set_ylabel('GF (K/(W/m^2))')
        ax.legend()
    
    if save_fig:
        if overlay:
            plt.savefig(f'{path_to_figures}{train_id}/mean_GFs.png', bbox_inches = 'tight', dpi = 350)
    
    return

def check_data(ds, train_id, single = False):
    
    for m in ds[train_id].keys():
        if single:
            ERF = ds
        else:
            ERF = ds[train_id][m]
            
        N_years = len(ERF['year'])
        offsets = [i for i in range(0,-N_years,-1)]
        input_matrix = diags(ERF.ERF.values,offsets=offsets,shape=(N_years,N_years),format='csr')
        array_mat = input_matrix.toarray()
        cond_num = LA.cond(array_mat)
        print(f'Model: {m}, Condition Number: {cond_num}')
        
    return

########################### CONVOLUTION AND COMPARISON ###################################

def import_regrid_tas(model_dict,run_id):
    tas = {}
    for m in model_dict.keys():
        print(f'{path_to_cmip6_data}tas_Amon_{model_dict[m]}')
        tas[m] = xr.open_dataset(f'{path_to_cmip6_data_clean}{run_id}/tas_Amon_{model_dict[m]}', use_cftime=True, chunks = 1000000)
        lat_corners = cfxr.bounds_to_vertices(tas[m].isel(time = 0)['lat_bnds'], "bnds", order=None)
        lon_corners = cfxr.bounds_to_vertices(tas[m].isel(time = 0)['lon_bnds'], "bnds", order=None)
        tas[m] = tas[m].assign(lon_b=lon_corners, lat_b=lat_corners)
        tas[m] = regrid_cmip(tas[m], ds_out)
        
    return tas

def import_regrid_tas_set(model_set, run_id, print_var = False):
    tas = {}
    for m in model_set:
        if print_var:
            print(f'{path_to_cmip6_data_clean}{run_id}/tas_Amon_{m}_{run_id}_r1i1p1f1.nc4')
        tas[m] = xr.open_dataset(f'{path_to_cmip6_data_clean}{run_id}/tas_Amon_{m}_{run_id}_r1i1p1f1.nc4', use_cftime=True, chunks = 1000000)
        lat_corners = cfxr.bounds_to_vertices(tas[m].isel(time = 0)['lat_bnds'], "bnds", order=None)
        lon_corners = cfxr.bounds_to_vertices(tas[m].isel(time = 0)['lon_bnds'], "bnds", order=None)
        tas[m] = tas[m].assign(lon_b=lon_corners, lat_b=lat_corners)
        tas[m] = regrid_cmip(tas[m], ds_out)
        
    return tas

def calc_tas_CMIP_set(tas_exp, tas_pictrl, model_set):
    tas_CMIP = {}
    for m in model_set:       
        # Remove climatology
        tas_CMIP[m] = tas_exp[m] - tas_pictrl[m].mean(dim = ['time'])
        tas_CMIP[m] = tas_CMIP[m].drop('height')

        # Time stamping only available up to 3000 months, so we limit that here
        if len(tas_CMIP[m]['time']) > 3000:
            periods = 3000
        else:
            periods = len(tas_CMIP[m]['time'])

        times = pd.date_range('1850', periods = periods, freq='MS')
        weights = times.shift(1, 'MS') - times
        weights = xr.DataArray(weights, [('time', tas_CMIP[m]['time'][:periods].values)]).astype('float')
        tas_CMIP[m] = (tas_CMIP[m] * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')

        # Start from t = 0, such that all model times match
        tas_CMIP[m]['year'] = range(len(tas_CMIP[m]['year']))
    
    tas_CMIP_ds = concat_multirun(tas_CMIP,'model')
    tas_CMIP_ds = tas_CMIP_ds.rename({'year':'s'})
    
    return tas_CMIP_ds

def convolve_exp(G_ds, RF_ds, model_dict, train_id, expmean = True):
    if conv_mean:
        GF = G_ds.weighted(A).mean(dim = ['lat','lon'])
    else:
        GF = G_ds

    conv = {}
    for m1 in model_dict.keys():
        conv[m1] = {}
        if conv_mean:
            print(f'Convolving {m1} for Global Mean')
        else:
            print(f'Convolving {m1} Spatially')
        for train in train_id:
            # Relabeled realizations to simplify syntax
            try:
                m2 = m1.split('_')[0]

            except:
                m2 = m1
                
            # might need to add back: train_ids = train
            if conv_mean:
                conv[m1][train] = signal.convolve(np.array(GF.sel(model = m2, train_id = train).dropna(dim = 's')),
                                                np.array(RF_ds.sel(model = m1)['RF']),'full')
                conv[m1][train] = np_to_xr_mean(conv[m1][train], GF.sel(model = m2, train_id = train), RF_ds.sel(model = m1))
                length = max(len(GF.dropna(dim = 's')['s']),len(np.array(RF_ds.sel(model = m1)['RF'])))
                conv[m1][train] = conv[m1][train][:length]
            
            else:
                conv[m1][train] = signal.convolve(np.array(GF.sel(model = m2, train_id = train).dropna(dim = 's')), 
                                           np.array(RF_ds.sel(model = m1)['RF'])[~np.isnan(np.array(RF_ds.sel(model = m1)['RF']))][..., None, None],
                                           'full')
                conv[m1][train] = np_to_xr(conv[m1][train], GF.sel(model = m2, train_id = train), RF_ds.sel(model = m1))
    
    conv_dict = {}
    for m in conv.keys():
        conv_dict[m] = concat_multirun(conv[m],'train_id')
    conv_ds = concat_multirun(conv_dict,'model')

    return conv_ds

def convolve_exp_meanGF(G_ds, ERF_ds, train, conv_mean = True):
    if conv_mean:
        G_ds = G_ds.weighted(A).mean(dim = ['lat','lon'])
        #GF = G_ds.mean(dim = ['model'])
    #else:
        #GF = G_ds.mean(dim = ['model'])

    GF = G_ds
    ERF_ds = ERF_ds.mean(dim = ['model'])
        
    conv = {} 
    if conv_mean:
        print(f'Convolving mean GF for Global Mean')
        conv[train] = signal.convolve(np.array(GF.dropna(dim = 's')),
                                            np.array(ERF_ds['ERF']),'full')
        conv[train] = np_to_xr_mean(conv[train], GF, ERF_ds)

    else:
        print(f'Convolving mean GF Spatially')
        conv[train] = signal.convolve(np.array(GF.dropna(dim = 's')), 
                                       np.array(ERF_ds['ERF'])[~np.isnan(np.array(ERF_ds['ERF']))][..., None, None],
                                       'full')
        conv[train] = np_to_xr(conv[train], GF, ERF_ds)
        
    length = max(len(GF.dropna(dim = 's')['s']),len(np.array(ERF_ds['ERF'])))
    conv[train] = conv[train][:length]
    
    conv_ds = concat_multirun(conv,'train_id')

    return conv_ds

def convolve_exp_set(G_ds, RF_ds, model_set, train_id, conv_mean = True):
    if conv_mean:
        GF = G_ds.weighted(A).mean(dim = ['lat','lon'])
    else:
        GF = G_ds

    conv = {}
    for m1 in model_set:
        conv[m1] = {}
        if conv_mean:
            print(f'Convolving {m1} for Global Mean')
        else:
            print(f'Convolving {m1} Spatially')
        for train in train_id:
            m2 = 'NorESM-LM'
            
            # might need to add back: train_ids = train
            if conv_mean:
                print(m2)
                conv[m1][train] = signal.convolve(np.array(GF.sel(train_id = train).dropna(dim = 's')),
                                                np.array(RF_ds.sel(model = m1)['RF']),'full')
                conv[m1][train] = np_to_xr_mean(conv[m1][train], GF.sel(train_id = train), RF_ds.sel(model = m1))
                length = max(len(GF.dropna(dim = 's')['s']),len(np.array(RF_ds.sel(model = m1)['RF'])))
                conv[m1][train] = conv[m1][train][:length]
            
            else:
                conv[m1][train] = signal.convolve(np.array(GF.sel(train_id = train).dropna(dim = 's')), 
                                           np.array(RF_ds.sel(model = m1)['RF'])[~np.isnan(np.array(RF_ds.sel(model = m1)['RF']))][..., None, None],
                                           'full')
                conv[m1][train] = np_to_xr(conv[m1][train], GF.sel(train_id = train), RF_ds.sel(model = m1))
                
    conv_dict = {}
    for m in conv.keys():
        conv_dict[m] = concat_multirun(conv[m],'train_id')
    conv_ds = concat_multirun(conv_dict,'model')

    return conv_ds
    
def plot_ERF_profile(ERF_ds, conv_id, model_color, save_fig = False):
    fig, ax = plt.subplots(figsize = [10,6])
    for m in ERF_ds.model:
        ax.plot(ERF_ds['year'],ERF_ds.sel(model = m)['ERF'], alpha = .8, label = f'{m.values}')#, color = model_color[str(m.values)])
    ax.plot(ERF_ds['year'],ERF_ds.mean(dim = 'model')['ERF'], color = 'k', label = f'Ensemble Mean')
    
    ax.legend()
    ax.set_xlabel('Year', fontsize = 14)
    ax.set_ylabel('$W/m^2$', fontsize = 14)
    ax.set_title(f'{conv_id} $F$ Profile')
    
    if save_fig:
        plt.savefig(f'{path_to_figures}{conv_id}/F_profiles_{conv_id}_placeholder.pdf', bbox_inches = 'tight', dpi = 350)
        
def plot_conv(train_id, conv_id, conv_mean_ds, ds_dif, type_color, save_fig = False):
    
    fig, ax = plt.subplots(figsize = [10,6])
    # Might need to add back in model weights, train_type
    for train in train_id:
        #ax.plot(conv_mean_ds.mean(dim = 'model').sel(train_id = train), label = names[train], linestyle = ':')
        
        for m in conv_mean_ds.model:
            name = str(m.values)
            ax.plot(conv_mean_ds['s'] + 1850, conv_mean_ds.sel(train_id = train, model = m), label = name, linestyle = ':')#,color = model_color[name])
        
    ax.plot(conv_mean_ds['s'] + 1850, conv_mean_ds.mean(dim = ['model', 'train_id']), label = f'Emulator', color = type_color['all'], linestyle = '--')
    ax.plot(np.arange(1850,1850 + len(ds_dif['s'])), ds_dif.mean(dim = 'model').weighted(A).mean(dim = ['lat','lon'])['tas'], 
         label = 'Ensemble Mean', color = type_color['model'])
    
    ax.legend()
    ax.set_xlabel('Years', fontsize = 14)
    ax.set_ylabel('$\Delta$T ($\degree$C)', fontsize = 14)
    ax.set_title(f'{train} Convolution with {conv_id}')
    
    if save_fig:
        plt.savefig(f'{path_to_figures}{conv_id}/global_conv_{conv_id}_placeholder.png', bbox_inches = 'tight', dpi = 350)
        
    return

def plot_conv_meanGF(train_id, conv_id, conv_mean_ds, tas_CMIP, save_fig = False):
    conv_id_cap = {'1pctCO2':'1pctCO2',
               'ssp126':'SSP126',
               'ssp245':'SSP245',
               'ssp370':'SSP370',
               'ssp585':'SSP585'}
    
    plt.style.use('seaborn-v0_8-colorblind')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    fig, ax = plt.subplots(figsize = [8,6])
    ax.plot(conv_mean_ds['s'] + 1850, conv_mean_ds.sel(train_id = train_id), label = 'Emulator Ensemble Mean', linewidth = 3)
    ax.plot(np.arange(1850,1850 + len(tas_CMIP['s'])), tas_CMIP.mean(dim = 'model').weighted(A).mean(dim = ['lat','lon'])['tas'], 
         label = 'CMIP6 Ensemble Mean', linewidth = 3, linestyle = '--',color=colors[2])
    
    ax.legend(fontsize = 18)
    ax.set_xlabel('Year', fontsize = 20)
    ax.set_ylabel('$\overline{\Delta T}\,(t)$ [$\degree$C]', fontsize = 20)
    ax.set_title(f'Global Mean Temperature Emulation\nPredictor: {train_id}, Target: {conv_id_cap[conv_id]}',fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.grid(True)
    fig.tight_layout()
    
    if save_fig:
        plt.savefig(f'{path_to_figures}conv_global_{train_id}_{conv_id}_v3.pdf', bbox_inches = 'tight', dpi = 350)
        
    return

def plot_conv_meanGF_sens(train_id, conv_id, conv_mean_ds, conv_mean_ds_plu10, conv_mean_ds_min10, tas_CMIP, save_fig = False):
    conv_id_cap = {'1pctCO2':'1pctCO2',
               'ssp126':'SSP126',
               'ssp245':'SSP245',
               'ssp370':'SSP370',
               'ssp585':'SSP585'}
    
    fig, ax = plt.subplots(figsize = [10,6])
    ax.plot(conv_mean_ds['s'] + 1850, conv_mean_ds.sel(train_id = train_id), label = 'Emulator Ensemble Mean', linewidth = 3,color=present_colors['blue'],path_effects=[pe.Stroke(linewidth=3.25, foreground='k'), pe.Normal()])
    ax.fill_between(conv_mean_ds['s'] + 1850, conv_mean_ds_min10.sel(train_id = train_id), conv_mean_ds_plu10.sel(train_id = train_id),color=present_colors['blue'],alpha=0.5)
    ax.plot(np.arange(1850,1850 + len(tas_CMIP['s'])), tas_CMIP.mean(dim = 'model').weighted(A).mean(dim = ['lat','lon'])['tas'], 
         label = 'CMIP6 Ensemble Mean', linewidth = 3, linestyle = '--', color=present_colors['orange'],path_effects=[pe.Stroke(linewidth=3.25, foreground='k'), pe.Normal()])

    ax.legend(fontsize = 18)
    ax.set_xlabel('Year', fontsize = 20)
    ax.set_ylabel('$\Delta \overline{T}$ ($\degree$C)', fontsize = 20)
    ax.set_title(f'Global Mean Temperature Emulation\nPredictor: {train_id}, Target: {conv_id_cap[conv_id]}',fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    fig.tight_layout()
    
    if save_fig:
        plt.savefig(f'{path_to_figures}conv_global_{train_id}_{conv_id}_sens_v1.pdf', bbox_inches = 'tight', dpi = 350)
        
    return

def plot_conv_enROADS(train_id, conv_id, conv_mean_ds, tas_combined, type_color, save_fig = False):
    fig,ax = plt.subplots(figsize = [10,6])
    for train in train_id:
            #ax.plot(conv_mean_ds.mean(dim = 'model').sel(train_id = train), label = names[train], color = type_color[train], linestyle = ':')
            
            for m in conv_mean_ds.model:
                name = str(m.values)
                ax.plot(conv_mean_ds['s'] + 1850, conv_mean_ds.sel(train_id = train, model = m), label = name, linestyle = ':')

    ax.plot(conv_mean_ds['s'] + 1850,conv_mean_ds.mean(dim = ['model', 'train_id']), label = f'Emulator', color = type_color['all'], linestyle = '--')
    ax.plot(np.arange(1850,1850 + len(tas_combined[name]['year'])), tas_combined[name]['tas'], label = 'EnROADS', color = type_color['model'])
    
    ax.legend()
    ax.set_xlabel('Years', fontsize = 14)
    ax.set_ylabel('$\Delta$T ($\degree$C)', fontsize = 14)
    ax.set_title(f'{conv_id} Convolution with {train}')
    if save_fig:
        plt.savefig(f'{path_to_figures}{conv_id}/global_conv_{conv_id}_placeholder.png', bbox_inches = 'tight', dpi = 350)
    
    return
        
def plot_dif_map(conv_ds, ds_dif, plot_yr, yr_dif, conv_id, dif = True, save_fig = False):
    plot_yr = plot_yr - 1850
    
    cmap = mpl.cm.RdBu_r
    #levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2.0]
    levels = np.linspace(-3.75,3.75,num = 2*8 + 1)
    vmin = -2
    vmax = 2
    fig, ax= plt.subplots(figsize = [10,6], subplot_kw = {'projection':ccrs.PlateCarree()}, constrained_layout = True)

    # Contours of difference
    if dif:
        (conv_ds -  ds_dif['tas']).mean(dim = 'model').mean(dim = 'train_id').sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's').plot(ax = ax, 
                                                                                                                      cmap = cmap,
                                                                                                                      levels = levels,
                                                                                                                      extend = 'both', 
                                                                                                                      add_colorbar = True,     
                                                                                                                      transform = ccrs.PlateCarree(),
                                                                                                                      cbar_kwargs = {'label':r'$\Delta \overline{T}$ ($\degree$C)'})
    else:
        conv_ds.mean(dim = 'model').mean(dim = 'train_id').sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's').plot(ax = ax, 
                                                                                                                  cmap = mpl.cm.Reds, extend = 'both', 
                                                                                                                  add_colorbar = True,     
                                                                                                                  transform = ccrs.PlateCarree(),
                                                                                                                  cbar_kwargs = {'label':r'$\Delta \overline{T}$ ($\degree$C)'})
    if dif:
        ax.set_title(f'{conv_id}: Difference at {1850 + plot_yr} ($\pm {yr_dif}$) years', fontsize = 14)
    else:
        ax.set_title(f'{conv_id}: Temperature Change Relative to 1850 at {1850 + plot_yr} ($\pm {yr_dif}$) years', fontsize = 14)
    ax.coastlines()
    
    if save_fig:
        plt.savefig(f'{path_to_figures}{conv_id}/spatial_dif_{conv_id}_placeholder.png', bbox_inches = 'tight', dpi = 350)
        
    return

def plot_dif_map_meanGF(conv_ds, tas_CMIP, plot_yr, yr_dif, conv_id, train_id, dif = True, save_fig = False):
    conv_id_cap = {'1pctCO2':'1pctCO2',
               'ssp126':'SSP126',
               'ssp245':'SSP245',
               'ssp370':'SSP370',
               'ssp585':'SSP585'}
    
    plot_yr = plot_yr - 1850
    
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    cmap = mpl.cm.RdBu_r
    fig, ax= plt.subplots(figsize = [10,6], subplot_kw = {'projection':ccrs.Robinson()}, constrained_layout = True)

    tas_CMIP = tas_CMIP.mean(dim = 'model')
    
    # Contours of difference
    if dif:
        extremes = (-2, 2)
        norm = plt.Normalize(*extremes)
        im = (conv_ds -  tas_CMIP['tas']).mean(dim = 'train_id').sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's').plot(ax = ax, 
                                                                                                                      cmap = cmap,
                                                                                                                      norm = norm,   
                                                                                                                      transform = ccrs.PlateCarree())
    
    else:
        extremes = (0, 9)
        norm = plt.Normalize(*extremes)
        im = tas_CMIP['tas'].sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's').plot(ax = ax, 
                                                                                                 cmap = mpl.cm.Reds,
                                                                                                 norm = norm,
                                                                                                 transform = ccrs.PlateCarree()) 
        #im = conv_ds.mean(dim = 'train_id').sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's').plot(ax = ax, 
        #                                                                                                          cmap = mpl.cm.Reds,     
        #                                                                                                          transform = ccrs.PlateCarree()) 
    #if dif:
    #    ax.set_title(f'{conv_id}: Difference at {1850 + plot_yr} ($\pm {yr_dif}$) years', fontsize = 14)
    #else:
    #    ax.set_title(f'{conv_id}: Temperature Change Relative to 1850 at {1850 + plot_yr} ($\pm {yr_dif}$) years', fontsize = 14)
    
    #cb = fig.colorbar(im)
    #cb.remove()
    
    ax.coastlines()
    cb = fig.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.7, extend = 'both')
    cb.set_label('$\Delta T$ [$\degree$C]', fontsize = 20)
    cb.ax.tick_params(labelsize=16)
    
    if dif:
        ax.set_title(f'(Emulator - CMIP6) in {plot_yr + 1850} ($\pm {yr_dif}$ years)\nPredictor: {train_id}, Target: {conv_id_cap[conv_id]}', fontsize = 20)
    else:
        ax.set_title(f'Temperature Change Relative to 1850 in {plot_yr + 1850} ($\pm {yr_dif}$ years)\nCMIP6 Scenario: {conv_id_cap[conv_id]}', fontsize = 20)
    #fig.tight_layout()
    
    if save_fig:
        if dif:
            plt.savefig(f'{path_to_figures}conv_spatial_dif_{train_id}_{conv_id}_{plot_yr}_v3.pdf', bbox_inches = 'tight', dpi = 350)
        else:
            plt.savefig(f'{path_to_figures}cmip_spatial_raw_{train_id}_{conv_id}_{plot_yr}_v3.pdf', bbox_inches = 'tight', dpi = 350)
        
    return

def plot_pattern(pattern_ds, train_id, test_id = None, tas_CMIP = None, plot_yr = None, yr_dif = None, save_fig = False):
    cmap = mpl.cm.RdBu_r
    if tas_CMIP: 
        plot_yr = plot_yr - 1850
        levels = np.linspace(-0.5,0.5,num = 23)
    else:
        levels = np.linspace(-0.5,2.5,num = 2*10 + 1)
    vmin = -2
    vmax = 2
    fig, ax= plt.subplots(figsize = [10,6], subplot_kw = {'projection':ccrs.Robinson()}, constrained_layout = True)
    
    if tas_CMIP:
        (tas_CMIP.mean(dim = 'model') - pattern_ds).sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's')['tas'].plot(ax = ax, 
                                            cmap = cmap, 
                                            levels = levels,
                                            extend = 'both', 
                                            add_colorbar = True,     
                                            transform = ccrs.PlateCarree(),
                                            cbar_kwargs = {'label':r'$\Delta \overline{T}$ ($\degree$C)'})
        
    else:
        pattern_ds['pattern'].plot(ax = ax,
                               cmap = cmap, 
                               levels = levels,
                               extend = 'both', 
                               add_colorbar = True,
                               transform = ccrs.PlateCarree(),
                               cbar_kwargs = {'label':r'$\Delta \overline{T}$ ($\degree$C)'})

    ax.coastlines()
    
    if tas_CMIP:
        ax.set_title(f'{train_id}_{test_id}: Difference at {1850 + plot_yr} ($\pm {yr_dif}$) years', fontsize = 14)
    else:
        ax.set_title(f'{train_id}: Pattern', fontsize = 14)
    ax.coastlines()
    
    #if save_fig:
    #    plt.savefig(f'{path_to_figures}{conv_id}/spatial_dif_{conv_id}_placeholder.png', bbox_inches = 'tight', dpi = 350)
        
    return #(pattern_ds).sel(s = slice(plot_yr-yr_dif, plot_yr+yr_dif)).mean(dim = 's')

def calc_error_metrics(truth, emulator, start_year, end_year, mean_GF = False, pattern = False):
    slice_start = start_year - 1850
    slice_end = end_year - 1850
    
    truth = truth.mean(dim = 'model')
    
    if mean_GF and pattern == False:
        emulator = emulator.mean(dim = 'model').mean(dim = 'train_id').sel(s = slice(min(truth.s),max(truth.s)))
    elif pattern == False:
        emulator = emulator.mean(dim = 'train_id').sel(s = slice(min(truth.s),max(truth.s)))
    
    truth = truth.sel(s = slice(slice_start,slice_end))
    emulator = emulator.sel(s = slice(slice_start,slice_end))
    
    if pattern:
        MSE = np.square(np.subtract(truth['tas'],emulator['tas'])).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        MSE = np.square(np.subtract(truth['tas'],emulator)).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    RMSE = math.sqrt(MSE)
    
    if mean_GF and pattern == False:
        MAE = np.abs(np.subtract(truth['tas'],emulator)).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s']).mean(dim = ['model'])
    elif pattern == False:
        MAE = np.abs(np.subtract(truth['tas'],emulator)).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        MAE = np.abs(np.subtract(truth['tas'],emulator['tas'])).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
        
    if pattern:
        bias = np.subtract(emulator['tas'],truth['tas']).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    else:
        bias = np.subtract(emulator,truth['tas']).weighted(A).mean(dim = ['lat','lon']).mean(dim = ['s'])
    
    return MSE.values, RMSE, MAE.values, bias.values

def calc_error_metrics_EnROADS(truth, emulator):
    truth = truth.mean(dim = 'model')
    truth = truth.rename({'year':'s'})
    emulator = emulator.mean(dim = 'model').mean(dim = 'train_id')
    emulator['s'] = emulator['s'] + 1850
    
    RMSE = np.square(truth['tas'] - emulator)
    RMSE = RMSE.mean(dim = ['s'])**(-1/2)
    
    MAE = np.abs(truth['tas'] - emulator).mean(dim = ['s'])
    
    bias = (emulator - truth['tas']).mean(dim = ['s'])
    
    return RMSE.values, MAE.values, bias.values

### Define output grid size
ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
        "lon": (["lon"], np.arange(0, 360, 1)),
        "lat_b": (["lat_b"], np.arange(-90.,91.,1.0)),
        "lon_b":(["lon_b"], np.arange(.5, 361.5, 1.0))
    }
)


#### function to find area of a grid cell from lat/lon ####
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

def diff_lists(list1, list2):
    return list(set(list1).symmetric_difference(set(list2)))  # or return list(set(list1) ^ set(list2))

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

######################## Dataset Dictionaries ###########################

lam_dict = {'ACCESS-CM2':0.67,
            'ACCESS-ESM1-5':0.68, 
            'CAMS-CSM1-0':1.71, 
            'CanESM5':0.64,
            'INM-CM4-8':1.42,
            'INM-CM5-0':1.49,
            'MIROC6':1.47,
            'MRI-ESM2-0':1.07,
            'NorESM2-LM':1.13}

model_set = set(['ACCESS-CM2',
                 'ACCESS-ESM1-5',
                 'CAMS-CSM1-0',
                 'CanESM5',
                 'INM-CM4-8',
                 'INM-CM5-0',
                 'MIROC6',
                 'MRI-ESM2-0',
                 'NorESM2-LM'])

model_test_set = set(['MIROC6','ACCESS-ESM1-5','CAMS-CSM1-0'])

################## Colors for Plotting ######################

type_color = {'model':'maroon', 'all':'darkcyan'} 

model_color = {'ACCESS':'pink',
               'CAMS':'darkgreen',
               'CanESM5':'orange',
               'CanESM5-1':'sienna',
               'MIROC':'purple',
               'NorESM':'blue'}

present_colors = {'orange':'#FFAB3F',
                  'blue':'#648efe'}
