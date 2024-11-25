import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from ERFutils import path_to_ERF_outputs as output_path
from ERFutils import A
from scipy.signal import convolve

def convert_and_round(lat, lon, lat_values, lon_values):
    # Convert longitude from [-180, 180] to [0, 360]
    if lon < 0:
        lon += 360

    # Round latitude and longitude to nearest available values
    nearest_lat = lat_values[np.abs(lat_values - lat).argmin()]
    nearest_lon = lon_values[np.abs(lon_values - lon).argmin()]

    return nearest_lat, nearest_lon

def load_conv_data():

  # Import ERF datasets
  path_to_hist = ' Streamlit/ERF_hist_En-ROADS.nc4'
  ERF_hist_ds = xr.open_dataset(path_to_hist)
  ERF_hist = ERF_hist_ds.ERF.values

  # Import Green's Functions
  G_ds_path = f'{output_path}GFs/G_1pctCO2_mean_ds.nc4'
  G_ds = xr.open_dataset(G_ds_path)['G[tas]']
  G_ds.name = 'G[tas]'
  G_ds = G_ds.rename({'year':'s'})

  return ERF_hist, G_ds

def conv_enROADS(G_ds, ERF_hist, ERF_enROADS, conv_local=False, lat = None, lon=None):
  ERF_all = np.concatenate((ERF_hist, ERF_enROADS))

  lat_values = G_ds.lat.values
  lon_values = G_ds.lon.values
  lat, lon = convert_and_round(lat, lon, lat_values, lon_values)

  if conv_local:
    G_ds = G_ds.sel(lat = lat, lon = lon)

  GF = G_ds

  tas_conv = convolve(np.array(GF.dropna(dim = 's')), ERF_all[~np.isnan(ERF_all)][..., None, None], 'full')
  tas_conv_Glenn = {'lat' : GF.lat.values,
                    'lon' : GF.lon.values,
                    'v' : tas_conv}

  return tas_conv_Glenn