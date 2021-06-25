""" Process a selection of adverse weather events

Code has been hacked together quickly (in 1 week) and not guaranteed to work as
expected - please double check everything before using seriously!

author: Group 2, Energy Climate Hackathon 2021
"""
import logging

import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import xarray as xr

from energy_functions_22 import calc_hdd_cdd, calc_demand, \
    alpha_demand_GB, convert_to_windpower, \
    country_wind_power, mask_xarray

from process_events import *


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)


def read_awe_return(event):
    """Read in Adverse Weather Scenarios for Future Electricity Systems"""
    # Data repository
    data_loc_scenarios = '/badc/deposited2021/adverse_met_scenarios_electricity/data/'
    # here we select the most extreme duration winter wind drought events, but
    # other adverse events can be specified
    fpath_scenarios = {s: data_loc_scenarios + 'winter_wind_drought/uk/' +
        f'return_period_1_in_{s}_years/duration/gwl12-4degC/'
        for s in [2, 5, 10, 20, 50, 100]}

    # Read in temperature, windspeed of each event, store in dict
    dataset_scenarios = {s: xr.open_mfdataset(s_path+f'event{event}/*.nc') for s, s_path in fpath_scenarios.items()}
    # convert temperature (K to C)
    for s in dataset_scenarios:
        dataset_scenarios[s]['t2m'] -= 273.15

    return dataset_scenarios


def main():
    # Load datasets
    logging.info('Read in Adverse Weather Events (Future Energy Scenarios)')
    event = 3
    dataset_scenarios = read_awe_return(event)
    logging.info('Load country mask')
    country_shapely = load_country_mask()

    # Apply country masks
    logging.info('Apply country mask')
    # Overwrite scenario datasets with masked data
    for s, ds in dataset_scenarios.items():
        logging.info(f'... scenario {s} ...')
        dataset_scenarios[s], mask_scenarios = mask_xarray(ds, country_shapely, ['t2m', 'wind_speed'], 1)

    # Aggregated variables
    logging.info('Aggregate variables to land average')
    ds_mean_scenarios = {s: land_average(ds) for s, ds in dataset_scenarios.items()}

    # Aggregated Temperature & Demand
    logging.info('Calculate demand (HDD/CDD model)')
    demand_scenarios = {s: apply_demand_model(ds) for s, ds in ds_mean_scenarios.items()}

    # Aggregated Windspeed & Windpower
    logging.info('Calculate windpower (physical based model)')
    for s, ds in dataset_scenarios.items():
        demand_scenarios[s]['windspeed'] = ds_mean_scenarios[s].wind_speed
        ds_daily = ds.resample(time='1D').mean('time')
        demand_scenarios[s]['windpower'] = \
            wind_conversion(ds_daily, mask_scenarios)

    # electricity residual load (GW)
    logging.info('Calculate electricity load')
    for s, ds in ds_mean_scenarios.items():
        demand_scenarios[s]['load'] = demand_scenarios[s].demand - demand_scenarios[s].windpower

    # Save data for further processing
    try:
        logging.info('Save data')
        for s, ds in demand_scenarios.items():
            fpath = f'data/ds_event{event}_return{s}.nc'
            ds.to_xarray().to_netcdf(fpath)
            print(f'Saved scenario {s} data to {fpath}')
    except:
        logging.warn('Failed to save, entering interactive mode...')
        import IPython
        IPython.embed()


if __name__ == '__main__':
    main()
