import numpy as np
import pandas as pd
import pooch

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

import xarray as xr
import scipy.signal as signal
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from ERFutils import path_to_ERF_outputs as output_path
from ERFutils import path_to_figures as figure_path
from ERFutils import A

def FaIR_ensemble():
    """
    Run the calibrated, constrained ensemble from FaIR.
    https://docs.fairmodel.net/en/latest/examples/calibrated_constrained_ensemble.html
    
    Args:
        None.
        
    Returns:
        f: The FaIR ensemble.
    """
    
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2300, 1)  # start, end, step
    f.define_scenarios(scenarios)

    fair_params_1_2_0_obj = pooch.retrieve(
        url = 'https://zenodo.org/record/8399112/files/calibrated_constrained_parameters.csv',
        known_hash = 'md5:de3b83432b9d071efdd1427ad31e9076',
    )

    df_configs = pd.read_csv(fair_params_1_2_0_obj, index_col=0)
    configs = df_configs.index  # this is used as a label for the "config" axis
    f.define_configs(configs)

    species, properties = read_properties(filename='FaIR_Data/species_configs_properties_calibration1.2.0.csv')
    f.define_species(species, properties)

    f.allocate()

    f.fill_from_rcmip()

    rcmip_emissions_file = pooch.retrieve(
        url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
        known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
    )
    df_emis = pd.read_csv(rcmip_emissions_file)
    gfed_sectors = [
        "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
        "Emissions|NOx|MAGICC AFOLU|Forest Burning",
        "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
        "Emissions|NOx|MAGICC AFOLU|Peat Burning",
    ]
    for scenario in scenarios:
        f.emissions.loc[dict(specie="NOx", scenario=scenario)] = (
            df_emis.loc[
                (df_emis["Scenario"] == scenario)
                & (df_emis["Region"] == "World")
                & (df_emis["Variable"].isin(gfed_sectors)),
                "1750":"2300",
            ]
            .interpolate(axis=1)
            .values.squeeze()
            .sum(axis=0)
            * 46.006
            / 30.006
            + df_emis.loc[
                (df_emis["Scenario"] == scenario)
                & (df_emis["Region"] == "World")
                & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
                "1750":"2300",
            ]
            .interpolate(axis=1)
            .values.squeeze()
            + df_emis.loc[
                (df_emis["Scenario"] == scenario)
                & (df_emis["Region"] == "World")
                & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
                "1750":"2300",
            ]
            .interpolate(axis=1)
            .values.squeeze()
        )[:550, None]
        
        solar_obj = pooch.retrieve(
        url = 'https://raw.githubusercontent.com/chrisroadmap/fair-add-hfc/main/data/solar_erf_timebounds.csv',
        known_hash = 'md5:98f6f4c5309d848fea89803683441acf',
    )

    volcanic_obj = pooch.retrieve(
        url = 'https://raw.githubusercontent.com/chrisroadmap/fair-calibrate/main/data/forcing/volcanic_ERF_1750-2101_timebounds.csv',
        known_hash = 'md5:c0801f80f70195eb9567dbd70359219d',
    )
    
    df_solar = pd.read_csv(solar_obj, index_col="year")
    df_volcanic = pd.read_csv(volcanic_obj)
    
    solar_forcing = np.zeros(551)
    volcanic_forcing = np.zeros(551)
    volcanic_forcing[:352] = df_volcanic.erf.values
    solar_forcing = df_solar["erf"].loc[1750:2300].values

    trend_shape = np.ones(551)
    trend_shape[:271] = np.linspace(0, 1, 271)
    
    fill(
        f.forcing,
        volcanic_forcing[:, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f.forcing,
        solar_forcing[:, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze()
        + trend_shape[:, None, None] * df_configs["fscale_solar_trend"].values.squeeze(),
        specie="Solar",
    )
    
    fill(f.climate_configs["ocean_heat_capacity"], df_configs.loc[:, "clim_c1":"clim_c3"].values)
    fill(
        f.climate_configs["ocean_heat_transfer"],
        df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], df_configs["clim_epsilon"].values.squeeze())
    fill(f.climate_configs["gamma_autocorrelation"], df_configs["clim_gamma"].values.squeeze())
    fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
    fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
    fill(f.climate_configs["seed"], df_configs["seed"])
    fill(f.climate_configs["stochastic_run"], True)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])
    
    f.fill_species_configs(filename='FaIR_Data/species_configs_properties_calibration1.2.0.csv')

    # carbon cycle
    fill(f.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_airborne"], df_configs["cc_rA"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_temperature"], df_configs["cc_rT"].values.squeeze(), specie="CO2")

    # aerosol indirect
    fill(f.species_configs["aci_scale"], df_configs["aci_beta"].values.squeeze())
    fill(f.species_configs["aci_shape"], df_configs["aci_shape_so2"].values.squeeze(), specie="Sulfur")
    fill(f.species_configs["aci_shape"], df_configs["aci_shape_bc"].values.squeeze(), specie="BC")
    fill(f.species_configs["aci_shape"], df_configs["aci_shape_oc"].values.squeeze(), specie="OC")

    # aerosol direct
    for specie in [
        "BC",
        "CH4",
        "N2O",
        "NH3",
        "NOx",
        "OC",
        "Sulfur",
        "VOC",
        "Equivalent effective stratospheric chlorine"
    ]:
        fill(f.species_configs["erfari_radiative_efficiency"], df_configs[f"ari_{specie}"], specie=specie)

    # forcing scaling
    for specie in [
        "CO2",
        "CH4",
        "N2O",
        "Stratospheric water vapour",
        "Contrails",
        "Light absorbing particles on snow and ice",
        "Land use"
    ]:
        fill(f.species_configs["forcing_scale"], df_configs[f"fscale_{specie}"].values.squeeze(), specie=specie)
    # the halogenated gases all take the same scale factor
    for specie in [
        "CFC-11",
        "CFC-12",
        "CFC-113",
        "CFC-114",
        "CFC-115",
        "HCFC-22",
        "HCFC-141b",
        "HCFC-142b",
        "CCl4",
        "CHCl3",
        "CH2Cl2",
        "CH3Cl",
        "CH3CCl3",
        "CH3Br",
        "Halon-1211",
        "Halon-1301",
        "Halon-2402",
        "CF4",
        "C2F6",
        "C3F8",
        "c-C4F8",
        "C4F10",
        "C5F12",
        "C6F14",
        "C7F16",
        "C8F18",
        "NF3",
        "SF6",
        "SO2F2",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-23",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-32",
        "HFC-365mfc",
        "HFC-4310mee",
    ]:
        fill(f.species_configs["forcing_scale"], df_configs["fscale_minorGHG"].values.squeeze(), specie=specie)

    # ozone
    for specie in ["CH4", "N2O", "Equivalent effective stratospheric chlorine", "CO", "VOC", "NOx"]:
        fill(f.species_configs["ozone_radiative_efficiency"], df_configs[f"o3_{specie}"], specie=specie)

    # initial value of CO2 concentration (but not baseline for forcing calculations)
    fill(
        f.species_configs["baseline_concentration"],
        df_configs["cc_co2_concentration_1750"].values.squeeze(),
        specie="CO2"
    )
    
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    
    f.run()
    
    return f

def plot_temp_FaIR(f):
    """
    Plots SSP temperature anomalies from FaIR ensemble.
    
    Args:
        f: The FaIR ensemble.
        
    Returns:
        None
    """
    
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i, scenario in enumerate(scenarios):
        for pp in ((0, 100), (5, 95), (16, 84)):
            ax[i // 4, i % 4].fill_between(
                f.timebounds,
                np.percentile(
                    f.temperature.loc[dict(scenario=scenario, layer=0)]
                    - np.average(
                        f.temperature.loc[
                            dict(scenario=scenario, timebounds=np.arange(1850, 1902), layer=0)
                        ],
                        weights=weights_51yr,
                        axis=0
                    ),
                    pp[0],
                    axis=1,
                ),
                np.percentile(
                    f.temperature.loc[dict(scenario=scenario, layer=0)]
                    - np.average(
                        f.temperature.loc[
                            dict(scenario=scenario, timebounds=np.arange(1850, 1902), layer=0)
                        ],
                        weights=weights_51yr,
                        axis=0
                    ),
                    pp[1],
                    axis=1,
                ),
                color=ar6_colors[scenarios[i]],
                alpha=0.2,
                lw=0
            )

        ax[i // 4, i % 4].plot(
            f.timebounds,
            np.median(
                f.temperature.loc[dict(scenario=scenario, layer=0)]
                - np.average(
                    f.temperature.loc[
                        dict(scenario=scenario, timebounds=np.arange(1850, 1902), layer=0)
                    ],
                    weights=weights_51yr,
                    axis=0
                ),
                axis=1,
            ),
            color=ar6_colors[scenarios[i]],
        )
    #     ax[i // 4, i % 4].plot(np.arange(1850.5, 2021), gmst, color="k")
        ax[i // 4, i % 4].set_xlim(1850, 2300)
        ax[i // 4, i % 4].set_ylim(-1, 10)
        ax[i // 4, i % 4].axhline(0, color="k", ls=":", lw=0.5)
        ax[i // 4, i % 4].set_title(fancy_titles[scenarios[i]])

    plt.suptitle("SSP temperature anomalies")
    fig.tight_layout()

    return

def plot_ERF_FaIR(f):
    """
    Plots SSP ERF anomalies from FaIR ensemble.
    
    Args:
        f: The FaIR ensemble.
        
    Returns:
        None
    """
    
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i, scenario in enumerate(scenarios):
        for pp in ((0, 100), (5, 95), (16, 84)):
            ax[i // 4, i % 4].fill_between(
                f.timebounds,
                np.percentile(
                    f.forcing_sum.loc[dict(scenario=scenario)],
                    pp[0],
                    axis=1,
                ),
                np.percentile(
                    f.forcing_sum.loc[dict(scenario=scenario)],
                    pp[1],
                    axis=1,
                ),
                color=ar6_colors[scenarios[i]],
                alpha=0.2,
                lw=0
            )

        ax[i // 4, i % 4].plot(
            f.timebounds,
            np.median(
                f.forcing_sum.loc[dict(scenario=scenario)],
                axis=1,
            ),
            color=ar6_colors[scenarios[i]],
        )
        ax[i // 4, i % 4].set_xlim(1850, 2300)
        ax[i // 4, i % 4].set_ylim(0, 15)
        ax[i // 4, i % 4].axhline(0, color="k", ls=":", lw=0.5)
        ax[i // 4, i % 4].set_title(fancy_titles[scenarios[i]])

    plt.suptitle("SSP effective radiative forcing")
    fig.tight_layout()
    
    return

def convolve_FaIR(f, G_ds, conv_mean=True, lat_lon = None, scenario_init=None):
    """
    Helper function to convolve the FaIR ensemble with the Green's functions.
    Required as the FaIR structure is different than the structure used for the 
    rest of the ERF datasets.
    
    Args:
        f: The FaIR ensemble.
        G_ds: Dataset containing the Green's functions.
        conv_mean: Whether or not to use the global mean GF.
        lat_lon: Tuple indicating spatial location as (lat,lon).
        scenario_init: Scenario to use in convolution.
        
    Returns:
        all_conv: Convolution data for a given location.
    
    """
    
    if conv_mean:
        G_ds = G_ds.weighted(A).mean(dim = ['lat','lon'])
        GF = G_ds

    else:
        GF = G_ds.sel(lat = lat_lon[0], lon = lat_lon[1])

    ERF_ds = f.forcing_sum.loc[dict(scenario=scenario_init)].sel(timebounds=slice(1850,2100))
    length = max(len(GF.dropna(dim = 's')['s']),len(np.array(ERF_ds)))

    all_conv = [signal.convolve(np.array(GF.dropna(dim = 's')),
                                np.array(ERF_ds.sel(config=conf)),
                                'full')[:length] for conf in ERF_ds.config]
    
    return all_conv

def run_spatial_ensemble(f, train_id, lat_lon, labels, scens, save_fig = False, time_run = False, scens_cap = None):
    """
    Runs spatially explicit ensemble based on FaIR ERF projections.
    
    Args:
        f: The FaIR ensemble.
        train_id: Green's function chosen for evaluation.
        lat_lon: List of lat/lon pairs.
        labels: Labels of locations for each lat/lon pair.
        scens: Scenarios to emulate (e.g. ssp126).
        save_fig: Whether or not to save the resultant plots.
        time_run: Whether or not to time each location.
        scens_cap: Capitalized scenario titles to use instead of the defaults (optional).
        
    Returns:
        all_conv: Dictionary of all convolution data.
                  WARNING: Adding more locations may cause this to become extremely large.
                  Use caution when modifying.
    """
    all_conv = {}
    
    # Import Green's Functions
    G_ds_path = f'{output_path}GFs/G_{train_id}_mean_ds.nc4'
    G_ds = xr.open_dataset(G_ds_path)['G[tas]']
    G_ds.name = 'G[tas]'
    G_ds = G_ds.rename({'year':'s'})
    
    fig, ax = plt.subplots(len(scens),len(lat_lon),figsize=(12, 9),sharey=True,sharex='col')
    
    # Iterate over scenarios of interest
    for j in range(len(scens)):
        scenario_init = scens[j]
        print(f'Scenario: {scenario_init}')
        
        all_conv[scenario_init] = {}

        # Iterate over locations of interest
        for i in range(len(lat_lon)):
            print(f'\tLocation: {labels[i]}')
            if time_run:
                start = time.time()
                
            if i == 0:
                all_conv[scenario_init][i] = convolve_FaIR(f,G_ds,conv_mean=True, lat_lon = None, scenario_init=scenario_init)
            else:
                all_conv[scenario_init][i] = convolve_FaIR(f,G_ds,conv_mean=False, lat_lon = lat_lon[i],scenario_init=scenario_init)
            
            if time_run:
                end = time.time()
                print(f'\t\tEnsemble run time: {end - start}')
            
            # Plot results
            for pp in ((0, 100), (5, 95), (16, 84), (50,50)):
                pct0 = np.percentile(all_conv[scenario_init][i],pp[0],axis=0)
                pct1 = np.percentile(all_conv[scenario_init][i],pp[1],axis=0)

                ax[j,i].fill_between(np.linspace(1850,1850 + len(all_conv[scenario_init][i][0])-1,len(all_conv[scenario_init][i][0])),
                                pct0,
                                pct1,
                                color=ar6_colors[scenario_init],
                                alpha=0.2,
                                lw=0)

                ax[j,i].plot(np.linspace(1850,1850 + len(all_conv[scenario_init][i][0])-1,len(all_conv[scenario_init][i][0])),
                             np.mean(all_conv[scenario_init][i],axis=0),
                             color=ar6_colors[scenario_init])

                ax[j,i].hlines(1.5, 1850, 2100, colors='r', linestyles='dashed', alpha = 0.6)
                ax[j,i].hlines(2, 1850, 2100, colors='k', linestyles='dashed', alpha = 0.6)

                ax[j,i].tick_params(axis='both', which='major', labelsize=18)

    rows = ['Row {}'.format(row) for row in ['A', 'B', 'C', 'D']]

    for axi, col in zip(ax[0], labels):
            axi.set_title(col, fontsize=18)
    
    if scens_cap:
        for axi, col in zip(ax[0], scens_cap):
            axi.set_title(col, fontsize=18)
    else:
        for axi, row in zip(ax[:,0], scens):
            axi.set_ylabel(row, fontsize=18)

    fig.supxlabel('Year',fontsize=20,x=0.525)
    fig.supylabel('$\Delta T$ [$^\circ$C]',fontsize=20)
    fig.suptitle('Probabilistic Temperature Forecast',fontsize=20,y=1.0,x=0.525)
    fig.tight_layout()
    
    if save_fig:
        plt.savefig(f'{figure_path}spatial_ensemble.pdf', bbox_inches = 'tight', dpi = 350)
    
    return all_conv

scenarios = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"]

fancy_titles = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp434": "SSP4-3.4",
    "ssp460": "SSP4-6.0",
    "ssp534-over": "SSP5-3.4-overshoot",
    "ssp585": "SSP5-8.5",
}

ar6_colors = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534-over": "#92397a",
    "ssp585": "#980002",
}

weights_51yr = np.ones(52)
weights_51yr[0] = 0.5
weights_51yr[-1] = 0.5