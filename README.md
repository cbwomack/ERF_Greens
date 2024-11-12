# Spatially Resolved Temperature Emulator
Calculates and evaluates the performance of spatially resolved temperature response functions (Green's functions) based on mapping ERF to temperature. <a href="https://essopenarchive.org/doi/full/10.22541/essoar.172124789.98568963">Our preprint describing the methodology within this repo can be found here.</a>

# Demo
We have built a small webapp to highlight the speed and predictive accuracy of our tool, <a href="https://bc3-erf-greens.streamlit.app/">which can be found here.</a> The tool allows users to emulate spatial temperature anomalies for 4 different scenarios using our Green's function methodology. Figure 1 shows an example of the emulator's spatial outputs, while Figure 2 shows the CMIP6 ERF data used as the input to the emulator. Finally, Figure 3 illustrates the emulator's predictive skill against the CMIP6 ground truth surface temperature data. 

# Usage
## Jupyter Notebooks and Code
All code for training and running the emulators is written in python, specifically within the Jupyter notebooks included in this repo. This repo requires a number of packages to function properly, and the included environment file (ERF_Greens.yml) contains all necessary packages to get up and running. <a href="https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment">See this tutorial for instructions on loading an environment from a .yml.</a>

Notebooks are sorted numerically as:<br>
<ol start="0">
  <li>Preprocessing - preprocesses CMIP6 data if beginning from raw data. Requires download of CMIP6 data separate from this repo.</li>
  <li>Training - trains emulators from CMIP6 data.</li>
  <li>Evaluation - evaluates emulator performance against CMIP6 scenarios and performs sensitivity analysis.</li>
  <li>Plotting - creates plots for paper describing this emulator.</li>
  <li>Application - applies emulator to the FaIR calibrated, constrained ensemble of projections.</li>
</ol>

<b>All Jupyter notebooks require ERFutils.py to function.</b> This file contains a suite of helper functions which include CMIP6 data regridding, emulator training, emulator evaluation, etc.

## File Structure
This repository is broken into several subfolders containing the data necessary for emulator training, as well as some of the outputs from actually running the emulator. 

<b>ERF_Emulator</b> - contains all functions necessary to diagnose GFs and run emulator.<br>
<b>ERF_Outputs</b> - contains output data, including processed ERF profiles, Green's functions, and patterns.<br>
<b>FaIR_Data</b> - contains data required for FaIR ensemble generation

<b>The code requires the subfolder 'ERF_Outputs/tas' to function properly.</b> <br>
This folder is roughly 5 Gb, and as such is too large to store in this repo, <a href="https://www.dropbox.com/scl/fo/j3lj5r3ikeyw34p4h705p/ADtk0ib2HcW0JV7BAOn9rBs?rlkey=cuozzzik7y0vtj9ecdwv3yiat&dl=0">but you can find it on my Dropbox</a>. Otherwise, you can manually preprocess the raw CMIP6 tas data yourself using the provided scripts and create the folder locally.

<a href="https://aims2.llnl.gov/search/cmip6/">Raw CMIP6 data can instead be downloaded from this link</a>, but this requires significant preprocessing prior to usage with our framework. All data is stored using the netcdf package. Preprocessing scripts are included, and the general methodology for preprocessing is as follows.
1. For each model and variable, open all files (recorded monthly) and average them to yearly values. Ensure all data begins at year zero (e.g. transform from 1850 to 0).
2. For each model, calculate yearly ERF from the raw CMIP6 radiation terms, <a href="https://esd.copernicus.org/articles/11/737/2020/">climate feedback parameter</a>, and yearly global average tas values.
3. Ensure all models are regridded onto a consistent grid.
4. Concatenate ERF and tas data into datasets containing all models.

### CMIP6 Experiments
We use the following experiments for emulator training and evaluation. <a href="https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_experiment_id.html">Full CMIP6 experiment descriptions can be found here.</a>
<table class="tg"><thead>
  <tr>
    <th class="tg-xam4">Experiment</th>
    <th class="tg-xam4">Usage</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-sg5v"><i>1pctCO2</i></td>
    <td class="tg-sg5v">training, evaluation</td>
  </tr>
  <tr>
    <td class="tg-sg5v"><i>historical</i></td>
    <td class="tg-sg5v">evaluation</td>
  </tr>
  <tr>
    <td class="tg-sg5v"><i>ssp126</i></td>
    <td class="tg-sg5v">evaluation</td>
  </tr>
  <tr>
    <td class="tg-0w8i"><i>ssp245</i></td>
    <td class="tg-0w8i">evaluation</td>
  </tr>
  <tr>
    <td class="tg-0w8i"><i>ssp370</i></td>
    <td class="tg-0w8i">evaluation</td>
  </tr>
  <tr>
    <td class="tg-0w8i"><i>ssp585</i></td>
    <td class="tg-0w8i">evaluation</td>
</tbody>
</table>

### CMIP6 Models
We use the following set of models for training our emulators:
<table class="tg"><thead>
  <tr>
    <th class="tg-xam4">Center</th>
    <th class="tg-xam4">Model</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-sg5v">CAMS</td>
    <td class="tg-sg5v">CAMS-CSM1-0</td>
  </tr>
  <tr>
    <td class="tg-sg5v">CCCma</td>
    <td class="tg-sg5v">CanESM5</td>
  </tr>
  <tr>
    <td class="tg-sg5v">CSIRO-ARCCSS</td>
    <td class="tg-sg5v">ACCESS-CM2</td>
  </tr>
  <tr>
    <td class="tg-0w8i">CSIRO</td>
    <td class="tg-0w8i">ACCESS-ESM1-5</td>
  </tr>
  <tr>
    <td class="tg-0w8i">INM</td>
    <td class="tg-0w8i">INM-CM4-8</td>
  </tr>
  <tr>
    <td class="tg-0w8i">INM</td>
    <td class="tg-0w8i">INM-CM5-0</td>
  </tr>
  <tr>
    <td class="tg-0w8i">MIROC</td>
    <td class="tg-0w8i">MIROC6</td>
  </tr>
  <tr>
    <td class="tg-0w8i">MRI</td>
    <td class="tg-0w8i">MRI-ESM2-0</td>
  </tr>
  <tr>
    <td class="tg-0w8i">NCC</td>
    <td class="tg-0w8i">NorESM2-LM</td>
  </tr>
</tbody>
</table>
