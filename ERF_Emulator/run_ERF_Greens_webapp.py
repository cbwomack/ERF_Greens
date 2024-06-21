"""
Webapp to host climate emulator. From anthropogenic greenhouse gases
 to surface temperature anomalies.

Call via
$ conda activate emcli
$ streamlit run run_ERF_Greens_pocket_webapp.py
"""

import pickle # store and load compressed data
import pandas as pd
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # plot_tas_annual_local_err_map
import matplotlib.colors as colors # plot_tas_annual_local_err_map
import xarray as xr
import ERFutils

import os
os.environ['ESMFMKFILE'] = '/Users/chriswomack/anaconda3/envs/research/lib/esmf.mk'

def load_data_replot():

    if 'G_ds' not in st.session_state:
        # Load Green's Function
        G_ds_path = 'Convolution Inputs/G_1pctCO2.nc4'
        st.session_state['G_ds'] = xr.open_dataset(G_ds_path)['G[tas]']
        st.session_state['G_ds'] = st.session_state['G_ds'].rename({'year':'s'})

    if 'convolved' not in st.session_state or st.session_state['convolved'] != st.session_state['scenario']:
        # Load ERF profile
        ERF_ds_path = f'Convolution Inputs/ERF_hist_{st.session_state['scenario']}.nc4'
        st.session_state['ERF_ds'] = xr.open_dataset(ERF_ds_path)

        # Convolve to get temperature projection
        st.session_state['conv_ds'] = ERFutils.convolve_exp_meanGF(st.session_state['G_ds'], st.session_state['ERF_ds'], '1pctCO2', conv_mean = False)
        st.session_state['convolved'] = st.session_state['scenario']

    extremes = (0, 10)
    norm = plt.Normalize(*extremes)

    # Create figure with cartopy projection
    st.session_state['fig'], st.session_state['axs'] = plt.subplots(figsize=(6,4),
        subplot_kw=dict(projection=ccrs.Robinson()),
        dpi=200)

    im = st.session_state['conv_ds'].mean(dim = 'train_id').sel(s = slice(st.session_state['year'] - 1850 - 10, st.session_state['year'] - 1850 + 10)).mean(dim = 's').plot(ax = st.session_state['axs'],
    transform = ccrs.PlateCarree(),
    cmap = cm.Reds,
    norm = norm)

    cb = st.session_state['fig'].colorbar(im, orientation="horizontal", pad=0.05, extend = 'both')
    cb.set_label('$\Delta T$ [$\degree$C]', fontsize = 14)
    cb.ax.tick_params(labelsize=12)

    return

# Function to update the value in session state
def clicked(button_sel):
    for button in st.session_state['clicked']:
        if button == button_sel:
            st.session_state['clicked'][button] = True
        else:
            st.session_state['clicked'][button] = False

if __name__ == "__main__":
    # Main function, which is run for every change on the webpage

    st.write("""
    # BC3: ERF -> Temperature
    """)

    st.write(f"""
    ##### Select year for temperature projection:
    """)

    # Create a slider to retrieve input year selection
    st.session_state['year'] = st.slider('Year',
                                            min_value=1860,
                                            max_value=2089,
                                            value=1860,
                                            label_visibility='collapsed')

    # Set up buttons to select scenario
    col1, col2, col3, col4 = st.columns(4)

    # Default to SSP245
    if 'scenario' not in st.session_state or 'clicked' not in st.session_state:
        st.session_state['scenario'] = 'SSP245'
        st.session_state['clicked'] = {1:False,2:True,3:False,4:True}

    with col1:
        if st.button('SSP126', on_click=clicked, args=[1]):
            st.session_state['scenario'] = 'SSP126'
    with col2:
        if st.button("SSP245", on_click=clicked, args=[2]):
            st.session_state['scenario'] = 'SSP245'
    with col3:
        if st.button('SSP370', on_click=clicked, args=[3]):
            st.session_state['scenario'] = 'SSP370'
    with col4:
        if st.button("SSP585", on_click=clicked, args=[4]):
            st.session_state['scenario'] = 'SSP585'

    st.write(f"##### Selected Scenario: {st.session_state['scenario']}")

    # Load data and plot
    load_data_replot()

    st.session_state['axs'].coastlines()
    st.pyplot(st.session_state['fig'])