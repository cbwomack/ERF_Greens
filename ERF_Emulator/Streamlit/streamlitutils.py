# Contains only dependencies necessary for streamlit webapp
from scipy import signal
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.colors as mcolors
import pickle as pkl

data_path = 'ERF_Emulator/Streamlit/'
with open(f'{data_path}A.pickle, 'rb') as f:
    A = pkl.load(f)

def convolve_exp_meanGF(G_ds, ERF_ds, train_id, conv_mean = True, verbose = True):
    """
    Convolves a given experiment ERF profile with a Green's function
    to get the temperature response.

    Args:
        G_ds: Green's function dataset.
        ERF_ds: ERF dataset.
        train_id: ID indicating training dataset.
        conv_mean: Convolve with the global mean or all locations globally.
        verbose: Whether to print the current state of the function.

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
        if verbose:
            print(f'Convolving mean GF for Global Mean')
        conv[train_id] = signal.convolve(np.array(GF.dropna(dim = 's')),
                                            np.array(ERF_ds['ERF']),'full')
        conv[train_id] = np_to_xr_mean(conv[train_id], GF, ERF_ds)

    else:
        if verbose:
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

brewer2_light_rgb = np.divide([(102, 194, 165),
                               (252, 141,  98),
                               (141, 160, 203),
                               (231, 138, 195),
                               (166, 216,  84),
                               (255, 217,  47),
                               (229, 196, 148),
                               (179, 179, 179),
                               (202, 178, 214)],255)
brewer2_light = mcolors.ListedColormap(brewer2_light_rgb)
