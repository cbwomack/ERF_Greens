"""
Webapp to host climate emulator. Maps from ERF to surface temperature anomalies.
Outline for webapp courtesy of Bjorn Lutjens (https://github.com/blutjens/)

Call via
$ conda activate gchp
$ streamlit run run_ERF_Greens_webapp.py
"""
import pickle as pkl
import pandas as pd
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import xarray as xr
import streamlitutils
from matplotlib.lines import Line2D
from streamlitutils import brewer2_light
import time
import path
import sys

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

A_path = 'Convolution Inputs/A.pickle'
with open(A_path, 'rb') as f:
    A = pkl.load(f)

def load_data_replot():

    # Only required on first run, load the Green's function dataset
    if 'G_ds' not in st.session_state:
        # Load Green's Function
        G_ds_path = 'Convolution Inputs/G_1pctCO2.nc4'
        st.session_state['G_ds'] = xr.open_dataset(G_ds_path)['G[tas]']
        st.session_state['G_ds'] = st.session_state['G_ds'].rename({'year':'s'})

    # Redo convolution each time the scenario is updated
    if 'convolved' not in st.session_state or st.session_state['convolved'] != st.session_state['scenario']:
        # Load ERF profile
        ERF_ds_path = f'Convolution Inputs/ERF_hist_{st.session_state['scenario']}.nc4'
        st.session_state['ERF_ds'] = xr.open_dataset(ERF_ds_path)

        # Load tas profile
        tas_ds_path = f'Convolution Inputs/tas_hist_{st.session_state['scenario']}.nc4'
        st.session_state['tas_ds'] = xr.open_dataset(tas_ds_path)

        # Convolve to get temperature projection
        st.session_state['start_time'] = time.time()
        st.session_state['conv_ds'] = streamlitutils.convolve_exp_meanGF(st.session_state['G_ds'], st.session_state['ERF_ds'], '1pctCO2', conv_mean = False)
        st.session_state['end_time'] = time.time()
        st.session_state['convolved'] = st.session_state['scenario']

        # Only required after the first time the convolution is executed
        if 'fig1' not in st.session_state:
            # Create figure with cartopy projection
            st.session_state['fig1'], st.session_state['axs1'] = plt.subplots(figsize=(6,8),
                subplot_kw=dict(projection=ccrs.Robinson()),
                dpi=300)

            # Plot spatially resolved temperature change relative to 1850
            extremes = (0, 10)
            norm = plt.Normalize(*extremes)

            # When the user updates the inputs, the title and data to display will need to be updated
            st.session_state['axs1'].set_title(f'Temperature Change Relative to 1850 in {st.session_state['year']} ($\pm {10}$ years)', fontsize = 16)
            st.session_state['data'] = st.session_state['conv_ds'].mean(dim = 'train_id').sel(s = slice(st.session_state['year'] - 1850 - 10,
                                                                                                        st.session_state['year'] - 1850 + 10)).mean(dim = 's')
            st.session_state['im'] = st.session_state['axs1'].pcolormesh(st.session_state['conv_ds'].lon,
                                                                         st.session_state['conv_ds'].lat,
                                                                         st.session_state['data'],
                                                                         transform=ccrs.PlateCarree(),
                                                                         cmap=cm.Reds,
                                                                         norm = norm)

            # The colorbar should never need to be updated again
            cb = st.session_state['fig1'].colorbar(st.session_state['im'], orientation="horizontal", pad=0.05, extend = 'max')
            cb.set_label('$\Delta T$ [$\degree$C]', fontsize = 20)
            cb.ax.tick_params(labelsize=16)

    # Update title to reflect current year and get new data to plot
    st.session_state['axs1'].set_title(f'Temperature Change Relative to 1850 in {st.session_state['year']} ($\pm {10}$ years)', fontsize = 16)
    st.session_state['data'] = st.session_state['conv_ds'].mean(dim = 'train_id').sel(s = slice(st.session_state['year'] - 1850 - 10, st.session_state['year'] - 1850 + 10)).mean(dim = 's')
    st.session_state['im'].set_array(st.session_state['data'].values.ravel())

    # Plot ERF profile for chosen scenario
    st.session_state['fig2'], st.session_state['axs2'] = plt.subplots(figsize=(6,5), dpi=300)
    i = 0
    for m in st.session_state['ERF_ds'].model:
        st.session_state['axs2'].plot(st.session_state['ERF_ds']['year'][164:] + 1850,
                                      st.session_state['ERF_ds'].sel(model = m)['ERF'][164:],
                                      color = brewer2_light(0),
                                      alpha = .6,
                                      label = f'{m.values}')
        i += 1
    st.session_state['axs2'].plot(st.session_state['ERF_ds']['year'][164:] + 1850,
                                  st.session_state['ERF_ds'].mean(dim = 'model')['ERF'][164:],
                                  color = 'k',
                                  label = 'Ensemble Mean',
                                  linewidth=3)

    legend_elements = [Line2D([0], [0], color=brewer2_light(0), lw=3, label='Ensemble Member'),
                   Line2D([0], [0], color='k', lw=3, label='Ensemble Mean'),
                   Line2D([], [], marker='.', color='r', markersize=15,linestyle='None',label='Current Year')]

    st.session_state['axs2'].legend(handles=legend_elements,fontsize=14)
    st.session_state['axs2'].set_xlabel('Year', fontsize = 20)
    st.session_state['axs2'].set_ylabel('ERF [Wm$^{-2}$]', fontsize = 20)
    st.session_state['axs2'].set_title(f'ERF Profile: {st.session_state['scenario']}', fontsize = 20)
    st.session_state['axs2'].tick_params(axis='both', which='major', labelsize=16)

    st.session_state['axs2'].scatter(st.session_state['year'],
                                     st.session_state['ERF_ds'].mean(dim = 'model')['ERF'][st.session_state['year'] - 1850],
                                     facecolors='r',
                                     edgecolors='r',
                                     s=60,
                                     linewidths=1,
                                     zorder=3)

    # Plot tas profile for chosen scenario
    st.session_state['fig3'], st.session_state['axs3'] = plt.subplots(figsize=(6,5), dpi=300)
    i = 0
    for m in st.session_state['tas_ds'].model:
        st.session_state['axs3'].plot(st.session_state['tas_ds']['s'][164:] + 1850,
                                      st.session_state['tas_ds'].sel(model = m)['tas'][164:],
                                      color = brewer2_light(0),
                                      alpha = .6,
                                      label = f'{m.values}')
        i += 1
    st.session_state['axs3'].plot(st.session_state['tas_ds']['s'][164:] + 1850,
                                  st.session_state['tas_ds'].mean(dim = 'model')['tas'][164:],
                                  color = 'k',
                                  label = 'Ensemble Mean',
                                  linewidth=3)

    conv_data = st.session_state['conv_ds'].mean(dim = 'train_id').weighted(A).mean(dim = ['lat','lon'])
    st.session_state['axs3'].plot(conv_data['s'][164:] + 1850, conv_data[164:], ls='-.', color = brewer2_light(1), label = f'Ensemble Mean',linewidth=3)

    legend_elements = [Line2D([0], [0], color=brewer2_light(0), lw=3, label='Ensemble Member'),
                   Line2D([0], [0], color='k', lw=3, label='Ensemble Mean'),
                   Line2D([0], [0], color=brewer2_light(1), ls='-.', lw=3, label='Emulated Ensemble Mean'),
                   Line2D([], [], marker='.', color='r', markersize=15,linestyle='None',label='Current Year')]

    st.session_state['axs3'].legend(handles=legend_elements,fontsize=14)
    st.session_state['axs3'].set_xlabel('Year', fontsize = 20)
    st.session_state['axs3'].set_ylabel('$\overline{\Delta T}$ [$\degree$C]', fontsize = 20)
    st.session_state['axs3'].set_title(f'Global Mean Temperature: {st.session_state['scenario']}', fontsize = 20)
    st.session_state['axs3'].tick_params(axis='both', which='major', labelsize=16)

    st.session_state['axs3'].scatter(st.session_state['year'],
                                     st.session_state['tas_ds'].mean(dim = 'model')['tas'][st.session_state['year'] - 1850],
                                     facecolors='r',
                                     edgecolors='r',
                                     s=60,
                                     linewidths=1,
                                     zorder=3)

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
    ## Effective Radiative Forcing (ERF) to Surface Temperature Anomaly Emulator
    """)

    st.write("""
    This app provides a demo of a response function-based climate emulator which maps from ERF to surface temperature anomalies. Technical details for this emulator can be found at https://github.com/cbwomack/ERF_Greens/.
    """)

    st.write(f"""
    #### Select year for temperature projection:
    """)

    # Create a slider to retrieve input year selection
    st.session_state['year'] = st.slider('Year',
                                            min_value=2014,
                                            max_value=2089,
                                            value=2014,
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

    st.write(f"#### Selected Scenario: {st.session_state['scenario']}")

    # Load data and plot
    load_data_replot()
    st.write(f"Convolution required {np.round(st.session_state['end_time'] - st.session_state['start_time'],4)} seconds to run.")
    st.session_state['axs1'].coastlines()

    st.write("""
    #### Figure 1: Emulated surface temperature anomalies in user-selected year.
    Anomalies are shown relative to the pre-industrial control (1850 temperatures). The colorscale ranges from light to dark to indiciate low and high levels of warming, respectively. The colorbar is kept consistent across scenarios.
    """)
    st.pyplot(st.session_state['fig1'])

    st.write("""
    #### Figure 2: ERF time series data for scenario of interest.
    Outputs from individual CMIP6 models, the CMIP6 ensemble mean, and user-selected year are depicted as thin teal lines, a thick black line, and red dot, respectively.
    """)
    st.pyplot(st.session_state['fig2'])

    st.write("""
    #### Figure 3: Global mean temperature time series data for scenario of interest.
    Outputs from individual CMIP6 models, the CMIP6 ensemble mean, the emulated ensemble mean, and user-selected year are depicted as thin teal lines, a thick black line, a dot-dash orange line, and red dot, respectively.
    """)
    st.pyplot(st.session_state['fig3'])

    st.write("""
    ERF and ground-truth temperature data are taken from the CMIP6 archive, and details of the model-ensemble used in this work can be found at https://github.com/cbwomack/ERF_Greens/.
    """)

    st.write("""
    ##### This research was part of the Bringing Computation to the Climate Challenge (BC3) project and supported by Schmidt Sciences through the MIT Climate Grand Challenges.
    """)
