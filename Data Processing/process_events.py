""" Process a selection of adverse weather events, making comparison to ERA5
reanalysis / other datasets

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

import data_paths


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)


def read_awe(event_paths='winter_wind_drought/uk/most_extreme_events/duration/'):
    """Read in Adverse Weather Scenarios for Future Electricity Systems"""
    # here we select the most extreme duration winter wind drought events, but
    # other adverse events can be specified
    fpath_scenarios = {s: data_paths.awefes_data + event_paths + f'event{s}/' for s in [1,2,3]}

    # Read in temperature, windspeed of each event, store in dict
    dataset_scenarios = {s: xr.open_mfdataset(s_path+'*.nc') for s, s_path in fpath_scenarios.items()}
    # convert temperature (K to C)
    for s in dataset_scenarios:
        dataset_scenarios[s]['t2m'] -= 273.15

    return dataset_scenarios


def read_era5():
    """Read in ERA5 reanalysis weather data"""
    # Read in temperature, windspeed (and solar, ...)
    dataset_reanalysis = xr.open_mfdataset(data_paths.ear5_field_set_1+
                                           'ERA5_1hr_field_set_1_201*_*.nc')
    dataset_reanalysis['t2m'] -= 273.15 # conversion K to C
    dataset_reanalysis['wind_speed'] = np.sqrt(dataset_reanalysis.u100**2 + dataset_reanalysis.v100**2)

    # only keep relevant fields
    for k in ['u100', 'v100', 'u10', 'v10', 'mx2t', 'msl', 'mn2t', 'ssrd']:
        del dataset_reanalysis[k]

    return dataset_reanalysis


def load_country_mask(country_name='United Kingdom'):
    """Load UK Country Mask as a numpy polygon (compatible with xarray etc.)"""
    # loop through the countries and extract the appropriate shapefile
    countries_shp = shpreader.natural_earth(resolution='10m',
                                            category='cultural',
                                            name='admin_0_countries')

    # search for matching shapefile
    country_shapely = None
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes['NAME_LONG'] == country_name:
            logging.info(f'Found country mask: {country_name}')
            country_shapely = country.geometry
    return country_shapely


def land_average(dataset, dim=['longitude', 'latitude']):
    """Calculate land average of given dataset

    Note preloading should give some speed up 
    """
    dataset.load()
    return dataset.resample(time='1D').mean('time').mean(dim=dim)


def apply_demand_model(dataset):
    """Apply calc_hdd_cdd to the dataset"""
    # HDD / CDD model for daily electricity demand
    hdd, cdd = calc_hdd_cdd(dataset.t2m)
    demand = calc_demand(dataset.time, hdd, cdd, alpha_demand_GB)

    # return dataframe with variables
    data = pd.DataFrame({'demand': demand, 'hdd': hdd, 'cdd': cdd, 't2m': dataset.t2m})
    data['time'] = dataset.time
    return data.set_index('time')


def wind_conversion(dataset, mask):
    """Using windspeed data, calculate capacity factors and power"""
    # conversion data
    fpath_turbines = data_paths.wind_resources + 'United_Kingdom_ERA5_windfarm_dist.nc'
    fpath_power_curve = data_paths.wind_resources + 'Enercon_E70_2300MW_ECEM_turbine.csv'
    power_curve = pd.read_csv(fpath_power_curve, sep=' ', header=None,
                              names=['wind_speed', 'power', 'cf'])

    # apply conversion gridded windspeed to windpower
    gridded_windpower = convert_to_windpower(dataset.wind_speed,
                                             power_curve)

    # apply conversion to country windpower
    windpower, total_capacity = country_wind_power(gridded_windpower,
                                                   fpath_turbines,
                                                   dataset.latitude,
                                                   dataset.longitude)

    # scale for number of nans in mask
    # (yes I should have built into the convert_to_windpower function... todo!)
    total_capacity *= 1 / (1 - np.isnan(mask).mean())

    # windpower needs to be in GW (ie. convert from CF and from MW)
    return windpower * total_capacity * 1e-3


def main():
    # Load datasets
    logging.info('Read in Adverse Weather Events (Future Energy Scenarios)')
    dataset_scenarios = read_awe()
    logging.info('Read in ERA5 reanalysis')
    dataset_reanalysis = read_era5()
    logging.info('Load country mask')
    country_shapely = load_country_mask()

    # Apply country masks
    logging.info('Apply country mask')
    # Overwrite scenario datasets with masked data
    for s, ds in dataset_scenarios.items():
        logging.info(f'... scenario {s} ...')
        dataset_scenarios[s], mask_scenarios = mask_xarray(ds, country_shapely, ['t2m', 'wind_speed'], 1)

    # Overwrite reanalysis dataset with masked data
    logging.info('... reanalysis ...')
    dataset_reanalysis, mask_reanalysis = mask_xarray(dataset_reanalysis, country_shapely, ['t2m', 'wind_speed'], 1)
   
    # Aggregated variables
    logging.info('Aggregate variables to land average')
    logging.info('... reanalysis ...')
    ds_mean_reanalysis = land_average(dataset_reanalysis)
    logging.info('... scenarios ...')
    ds_mean_scenarios = {s: land_average(ds) for s, ds in dataset_scenarios.items()}

    # Aggregated Temperature & Demand
    logging.info('Calculate demand (HDD/CDD model)')
    logging.info('... reanalysis ...')
    demand_reanalysis = apply_demand_model(ds_mean_reanalysis)
    logging.info('... scenarios ...')
    demand_scenarios = {s: apply_demand_model(ds) for s, ds in ds_mean_scenarios.items()}

    # Aggregated Windspeed & Windpower
    logging.info('Calculate windpower (physical based model)')
    logging.info('... reanalysis ...')
    demand_reanalysis['windspeed'] = \
        ds_mean_reanalysis.wind_speed
    #wind_hourly = wind_conversion(dataset_reanalysis, mask_reanalysis)
    #cal_shape = (len(demand_reanalysis),
    #             len(wind_hourly)//len(demand_reanalysis))
    # calculate windpower hourly then aggregate to daily
    #demand_reanalysis['windpower_alt'] = \
    #    np.reshape(wind_hourly, cal_shape).mean(axis=1)
    # calculate windpower daily
    ds_daily = dataset_reanalysis.resample(time='1D').mean('time')
    demand_reanalysis['windpower'] = wind_conversion(ds_daily, mask_reanalysis)
    logging.info('... scenarios ...')
    for s, ds in dataset_scenarios.items():
        demand_scenarios[s]['windspeed'] = ds_mean_scenarios[s].wind_speed
        #wind_hourly = wind_conversion(ds, mask_scenarios)
        #cal_shape = (len(demand_scenarios[s]),
        #             len(wind_hourly)//len(demand_scenarios[s]))
        # calculate windpower hourly then aggregate to daily
        #demand_scenarios[s]['windpower_alt'] = \
        #    np.reshape(wind_hourly, cal_shape).mean(axis=1)
        # calculate windpower daily
        ds_daily = ds.resample(time='1D').mean('time')
        demand_scenarios[s]['windpower'] = \
            wind_conversion(ds_daily, mask_scenarios)

    # electricity residual load (GW)
    logging.info('Calculate electricity load')
    demand_reanalysis['load'] = demand_reanalysis.demand - demand_reanalysis.windpower
    for s, ds in ds_mean_scenarios.items():
        demand_scenarios[s]['load'] = demand_scenarios[s].demand - demand_scenarios[s].windpower

    # Save data for further processing
    try:
        logging.info('Save data')
        fpath = 'ds_reanalysis.nc'
        demand_reanalysis.to_xarray().to_netcdf(fpath)
        print(f'Saved reanalysis data to {fpath}')
        for s, ds in demand_scenarios.items():
            fpath = f'ds_scenario_{s}.nc'
            ds.to_xarray().to_netcdf(fpath)
            print(f'Saved scenario {s} data to {fpath}')
    except:
        logging.warn('Failed to save, entering interactive mode...')
        import IPython
        IPython.embed()


if __name__ == '__main__':
    main()
