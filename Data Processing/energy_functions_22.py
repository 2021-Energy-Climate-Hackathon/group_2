"""Energy Climate Hackathon June 2021

Weather-energy transfer functions and related code. Based on code by Hannah
Bloomfield 2021 h.c.bloomfield@reading.ac.uk (simplified from UREAD MERRA2
energy models) https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/met.1858 

Code has been hacked together quickly (in 1 week) and not guaranteed to work as
expected - please double check everything before using seriously!

Modified by J. Fallon - group 2
"""

import numpy as np
import shapely.geometry
import xarray as xr


# calc_hdd_cdd GB regression coefficients
# coefficients for other countries available from https://doi.org/10.1002/met.1858
alpha_demand_GB = [35.1, 0.0017, 0.75, np.nan, -0.81, np.nan, np.nan, np.nan, -0.90, 3.84, -4.69]


def mask_xarray(dataset, shape, keys, return_mask=False):
    """Mask an xarray dataset
    
    Parameters:
    ===========
        dataset (xarray.Dataset) : Dataset with lats, lons coords
        shape (shapely MultiPolygon) : Country or region mask
        keys (list or str) : dataset keys to apply max to
        return_mask (bool) : (optional) flag to return the mask array
    
    Returns:
    ========
        country_masked_data (xarray.Dataset) : Dataset with given keys masked
        (optional) mask (numpy.ndarray) : the region mask
    """
    # make grids of the lat and lon data
    LONS, LATS = np.meshgrid(dataset.longitude, dataset.latitude)
    # flatten lons, lats meshgrid for easier looping
    x, y = LONS.flatten(), LATS.flatten()
    
    # loop through all the lat/lon combinations to get the masked points
    # mask is nan by default, and 1.0 if selected
    MASK_MATRIX = np.zeros((len(x), 1)) + np.nan
    for i in range(0, len(x)):
        my_point = shapely.geometry.Point(x[i], y[i]) 
        if shape.contains(my_point): 
            MASK_MATRIX[i, 0] = 1.0 # creates 1s and 0s where the country is

    # reshape to grid
    MASK_MATRIX_RESHAPE = np.reshape(MASK_MATRIX, (dataset.latitude.size,
                                                   dataset.longitude.size))

    # now apply the mask to the data
    # copy dataset
    dataset_masked = dataset.copy()
    # ensure keys is a list
    if isinstance(keys, str):
        keys = [keys]
    # mask for each key
    for k in keys:
        dataset_masked[k] = dataset[k] * MASK_MATRIX_RESHAPE[None,]
    
    if return_mask:
        return dataset_masked, MASK_MATRIX_RESHAPE
    else:
        return dataset_masked


def calc_hdd_cdd(spatial_mean_t2m):
    """Take array of country_masked 2m temperature (celsius) and convert into a
    time series of heating-degree days (HDD) and cooling degree days (CDD)
    using the method from Bloomfield et al.,(2020)
    https://doi.org/10.1002/met.1858

    **Note** the function assumes daily temperatures (regression parameters
    won't work otherwise) 

    Parameters:
    ===========
        spatial_mean_t2m (array): array of country_masked 2m temperatures, Dimensions 
            [time, lat,lon] or [lat,lon] in units of celsius. 

    Returns:
    ========
        HDD_term (array): Dimesions [time], Timeseries of heating degree days
        CDD_term (array): Dimesions [time], Timeseries of cooling degree days
    """
    HDD_term = np.where(spatial_mean_t2m <= 15.5, 15.5 - spatial_mean_t2m, 0)
    CDD_term = np.where(spatial_mean_t2m >= 22.0, spatial_mean_t2m - 22.0, 0)
    return HDD_term, CDD_term


def calc_demand(calendar, hdd, cdd, coefficients):
    """Convert hdd, cdd, calendar into demand using the method from Bloomfield
    et al.,(2020) https://doi.org/10.1002/met.1858"""
    # any nan coefficients are set to 0
    a = np.nan_to_num(coefficients)
    
    # weekday coeffs
    # get weekday numbers calendar
    calendar_weekdays = calendar.to_series().dt.dayofweek
    # do they match day = 0, 1, 2, ...
    weekdays = [calendar_weekdays == w for w in range(7)]
    # multiply by respective cal coeffs
    week_coeffs = np.dot(a[4:], np.array(weekdays))
    
    return a[0] + a[1] + a[2] * hdd + a[3] * cdd + week_coeffs


def convert_to_windpower(wind_speed_data, power_curve):
    """Takes wind speeds at daily resolution, returns capactiy factor array

    Parameters:
    ===========
        gridded_wind_power (array): wind power capacity factor data, dimensions 
            [time,lat,lon]. Capacity factors range between 0 and 1.

        power_curve (DataFrame): The DataFrame containing the wind speeds
            (column 0, 'wind_speed') and capacity factors (column 2, 'cf') of
            the chosen wind turbine.

    Returns:
    ========
        wind_power_cf (array): Gridded wind Power capacity factor  
            data, dimensions [time,lat,lon]. Values vary between 0 and 1.
    """
    # first load in the power curve data
    power_curve_w = power_curve['wind_speed']
    power_curve_p = power_curve['cf']

    #interpolate to fine resolution.
    pc_winds = np.linspace(0,50,501)
    pc_power = np.interp(pc_winds, power_curve_w, power_curve_p)

    # apply power curve to flattend wind array
    reshaped_speed = np.array(wind_speed_data).flatten()
    # calculate bin indices for pc_winds array
    # (indexing starts from 1 so needs -1: 0 in the next bit to start from the lowest bin)
    test = np.digitize(reshaped_speed, pc_winds, right=False) 
    # make sure the bins don't go off the end (power is zero by then anyway)
    test[test == len(pc_winds)] = 500

    wind_power_flattened = 0.5*(pc_power[test-1] + pc_power[test])
    wind_power_cf = np.reshape(wind_power_flattened, np.shape(wind_speed_data))
    
    return wind_power_cf


def country_wind_power(gridded_wind_power, wind_turbine_locations, lats, lons):
    """Takes windpower or capacity factor, and weights by installed wind
    turbine locations.

    Parameters:
    ===========
        gridded_wind_power (array): wind power capacity factor data, dimensions 
            [time,lat,lon]. Capacity factors range between 0 and 1.

        wind turbine locations (str): The filename of a .nc file
            containing the amount of installed wind power capacity in gridbox

    Returns:
    ========
        wind_power_country_cf (array): Time series of wind Power capacity factor
            data, weighted by the installed capacity in each reanalysis
            gridbox from thewindpower.net database. dimensions [time]. 
            Values vary between 0 and 1.
    """
    # first load in the installed capacity data.
    dataset = xr.open_dataset(wind_turbine_locations)
    total_MW = dataset.totals
    total_capacity = float(np.sum(total_MW))
    
    # interpolate installed capacity to match gridded wind_power lats lons
    # normalised by sum of total MW
    total = np.array(total_MW.interp(lat=lats, lon=lons) / total_capacity)

    # New timeseries array
    #wind_power_country_cf = np.zeros_like(gridded_wind_power[0])

    #for i in range(0, len(wind_power_country_cf)):
    #    wind_power_country_cf[i] = np.sum(gridded_wind_power[i,:,:] * total)
    wind_power_country_cf = np.sum(np.array(gridded_wind_power) * total[None,], axis=(1,2))

    return wind_power_country_cf, total_capacity
