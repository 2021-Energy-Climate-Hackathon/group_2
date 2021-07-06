"""
Convert gridded daily 100m wind to regional average daily wind generation. Firstly cuts out region, calculates wind
capacity factor based on selected wind turbine power curve in each grid cell, applies installed capacity weighting,
sums this over the region and multiples by the national/regional level of installed wind, to give daily regional average
wind generation.

module load scitools/default-current

Project: NIC 2020
DePreSys version of WindGenModel.py - modified by Megan Pearce (original code review by Chris Kent).
Updated to use DePreSys location weightings, DePreSys turbine allocations and DePreSys estimated 100m wind.
"""

import argparse
import iris
from ascend import shape
import os
import stat
import numpy as np
import csv
import pandas as pd
import cartopy.io.shapereader as shpreader


def wind_dep_to_generation_csv(daily_wind_cube, shps, region, turbine_option, cap_scenario, enercon_file, gamesa_file,
                               vestas_file, ng_onshore_file, ng_offshore_file, cap_weights, final_shape_gb,
                               european_countries, total_weights, national_levels, ens, period):
    """
    Wind capacity factor in a grid box = wind turbine power curve (for assigned turbine type) applied to wind speed in
    that grid box (value between 0 and 1)
    Wind generation in a grid box = wind capacity factor x grid box weight x National level of installed wind / total
    weights in grid boxes in that country
    The grid box weights should sum to 1 for each country.
    """
    # Define the assigned turbines based on option 1 or 2
    turbine_file = ["/project/BG_Consultancy/NIC/data/turbine_map_1_depresys.nc",
                    "/project/BG_Consultancy/NIC/data/turbine_map_2_depresys.nc"]
    assigned_turbines = iris.load_cube(turbine_file[turbine_option - 1])
    assigned_turbines = assigned_turbines.data

    # Calculate power curve for each of the 5 wind turbine options
    gamesa_speed, gamesa_cf = read_power_curve(gamesa_file, ' ')
    vestas_speed, vestas_cf = read_power_curve(vestas_file, ' ')
    enercon_speed, enercon_cf = read_power_curve(enercon_file, ' ')
    ng_onshore_speed, ng_onshore_cf = read_power_curve(ng_onshore_file, ',')
    ng_offshore_speed, ng_offshore_cf = read_power_curve(ng_offshore_file, ',')

    # assign turbines based on turbine map option
    tmp = np.ma.masked_less(np.zeros(daily_wind_cube.shape), 1)  # create an array to hold the calculated values
    tmp[:, (assigned_turbines == 1)] = np.interp(daily_wind_cube.data[:, (assigned_turbines == 1)],
                                                 enercon_speed, enercon_cf)
    tmp[:, (assigned_turbines == 2)] = np.interp(daily_wind_cube.data[:, (assigned_turbines == 2)],
                                                 gamesa_speed, gamesa_cf)
    tmp[:, (assigned_turbines == 3)] = np.interp(daily_wind_cube.data[:, (assigned_turbines == 3)],
                                                 vestas_speed, vestas_cf)
    tmp[:, (assigned_turbines == 4)] = np.interp(daily_wind_cube.data[:, (assigned_turbines == 4)],
                                                 ng_onshore_speed, ng_onshore_cf)
    tmp[:, (assigned_turbines == 5)] = np.interp(daily_wind_cube.data[:, (assigned_turbines == 5)],
                                                 ng_offshore_speed, ng_offshore_cf)

    cf = tmp * cap_weights.data

    cf = iris.cube.Cube(cf, long_name='wind_capacity_factor', attributes=None,
                        dim_coords_and_dims=[(daily_wind_cube.coord('time'), 0),
                                             (daily_wind_cube.coord('latitude'), 1),
                                             (daily_wind_cube.coord('longitude'), 2)])

    # GB
    # mask capacity factor cube with GB shape, sum over lat and long
    wind_gen_cube_gb = final_shape_gb.mask_cube(cf).collapsed(['latitude', 'longitude'], iris.analysis.SUM)
    # Northern Ireland
    # mask capacity factor cube with GB shape, sum over lat and long
    final_shape_ni = shps.filter(region='Northern Ireland').unary_union()
    wind_gen_cube_ni = final_shape_ni.mask_cube(cf).collapsed(['latitude', 'longitude'], iris.analysis.SUM)

    # cap_scenario - chosen input for sensitivity
    # multiply region capacity factor with grid weights and installed capacity to get generation.
    gb_factor = national_levels[0, cap_scenario - 1] / total_weights[0]
    wind_gen_cube_wind_gen_cube_gb = iris.analysis.maths.multiply(wind_gen_cube_gb, gb_factor, dim=None,
                                                                  in_place=False)
    ni_factor = national_levels[1, cap_scenario - 1] / total_weights[1]
    wind_gen_cube_ni = iris.analysis.maths.multiply(wind_gen_cube_ni, ni_factor, dim=None, in_place=False)
    # Add GB and NI together
    wind_gen_cube_uk = wind_gen_cube_wind_gen_cube_gb.data + wind_gen_cube_ni.data

    # convert cube to dataframe ready to write to csv file, indexed by dates
    # extract dates from cube -  period variable
    dates = daily_wind_cube.coord('time').units.num2date(daily_wind_cube.coord('time').points)
    date_index = []
    for date in dates:
        date_index.append(date.strftime("%Y-%m-%d"))
    wind_gen_df = pd.DataFrame(wind_gen_cube_uk.data, index=date_index)
    wind_gen_df.columns = ['UK']

    os.makedirs(f"/data/users/mpearce/NIC/data/depresys/wind_generation/{ens}", exist_ok=True)
    # If region is UK, can save out wind_gen_df as is
    if region == 'United Kingdom':
        filename_csv = f"/data/users/mpearce/NIC/data/depresys/wind_generation/{ens}/WG_" \
                       + region.replace(" ", "_").lower() \
                       + '_turbine_option=' + str(turbine_option) \
                       + '_cap_scenario=' + str(cap_scenario) \
                       + f"_test_{period}.csv"
        with open(filename_csv, 'w') as csv_file:  # write to csv
            wind_gen_df.to_csv(path_or_buf=csv_file)
        os.chmod(f"{filename_csv}",
                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IXOTH | stat.S_IROTH)  # change permissions
        print(f"Save complete for ens{ens}")

    # if region is Europe, need to do the same for each European country and sum together with UK
    if region == 'Europe':
        # loop over European countries
        for i in range(len(european_countries)):
            final_shape = shps.filter(admin=european_countries[i]).unary_union()
            wind_gen_cube = final_shape.mask_cube(cf).collapsed(['latitude', 'longitude'], iris.analysis.SUM)

            factor = national_levels[i + 2, cap_scenario - 1] / total_weights[i + 2]
            wind_gen = iris.analysis.maths.multiply(wind_gen_cube, factor, dim=None, in_place=False)
            # Add a column to wind_gen_df for the given country
            wind_gen_df[european_countries[i]] = pd.DataFrame(wind_gen.data, index=date_index)
        # Sum over countries
        wind_gen_df["Total"] = wind_gen_df.sum(axis=1)
        # Save out all countries
        filename_csv = f"/data/users/mpearce/NIC/data/depresys/wind_generation/{ens}/WG_" \
                       + region.replace(" ", "_").lower() \
                       + '_turbine_option=' + str(turbine_option) \
                       + '_cap_scenario=' + str(cap_scenario) \
                       + f"_test_{period}.csv"
        with open(filename_csv, 'w') as csv_file:  # write to csv
            wind_gen_df.to_csv(path_or_buf=csv_file)
        os.chmod(f"{filename_csv}",
                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IXOTH | stat.S_IROTH)  # change permissions
        print(f"Save complete for ens{ens}")


def read_power_curve(fname, delim):
    x_speed = []
    y_cf = []
    with open(fname, 'rt') as csvfile:
        for row in csv.reader(csvfile, delimiter=delim):
            x_speed.append(float(row[0]))
            if not row[1]:
                y_cf.append(float(row[4]))
            else:
                y_cf.append(float(row[2]))
    return x_speed, y_cf


def country_total_weights(country_shape, weight_cube):
    """
    Divides weighting in all grid cells in a particular country by the sum of weightings in that country
    """
    masked_cube = country_shape.mask_cube(weight_cube)  # mask where is not the country being used
    country_sum = masked_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM)  # sum weightings across country
    return country_sum.data


def main():
    """
    files: directory and filename(s) of  DePreSys bias corrected mean daily estimated 100m wind speed
    regions: UK or Europe
    capacity weights: based on potential wind turbine locations from James Price research and natural earth urban areas
    wind turbine power curves: from UoR and National Grid files
    National levels: The total installed capacity of wind renewables in each country for each scenario we consider

    args_list:
    [0] = region
    [1] = turbine option (1 for UoR  assigned turbines in all European land grid boxes + NG in sea grid boxes and
            2 for NG onshore in UK, UoR assigned turbines in rest of European + NG in sea)
    [2] = capacity scenario (1 for current capacity, 2 for future capacity, 3 for future capacity *1.5
        # scenario 1: All current level
        # scenario 2: UK higher (120 GW in line with NG FES), other countries the same - only this one used.
        # scenario 3: All higher - UK 120, other countries increased proportionately to 600 GW total
    """
    # Pass in arguments from the .sh file - allows us to vary inputs for the sensitivity study
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    arg_list = [int(item) for item in args.list.split(',')]

    if arg_list[0] == 1:
        region = 'United Kingdom'
    else:
        region = 'Europe'

    # Input files:
    # Turbine power curve files
    enercon_file = "/project/BG_Consultancy/NIC/data/ERA5_reanalysis_models/" \
                   "wind_power_model_outputs/Enercon_E70_2300MW_turbine.csv"  # Type 1
    gamesa_file = "/project/BG_Consultancy/NIC/data/ERA5_reanalysis_models/" \
                  "wind_power_model_outputs/Gamesa_G87_2000MW_turbine.csv"  # Type 2
    vestas_file = "/project/BG_Consultancy/NIC/data/ERA5_reanalysis_models/" \
                  "wind_power_model_outputs/Vestas_v110_2000MW_turbine.csv"  # Type 3
    ng_onshore_file = "/project/BG_Consultancy/NIC/data/NG_PowerCurves_onshore.csv"  # Type 4
    ng_offshore_file = "/project/BG_Consultancy/NIC/data/NG_PowerCurves_offshore.csv"  # Type 5

    # load natural earth admin state provinces for filtering throughout script
    natural_earth = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    shps = shape.load_shp(natural_earth)

    # Weighting of UK potential wind turbine locations from uk_locations.py
    cap_weights_file = '/project/BG_Consultancy/NIC/data/joined_weights_wind_depresys.nc'
    cap_weights = iris.load_cube(cap_weights_file)
    weights = iris.analysis.cartography.area_weights(cap_weights, normalize=True)
    cap_weights.data = cap_weights.data * weights

    # Define GB shape = GB land + offshore
    offshore = '/project/BG_Consultancy/NIC/data/turbine_location_shapefiles/offshore_buildable_areas_gr_5.shp'
    spheroid = iris.coord_systems.GeogCS(semi_major_axis=6377563.396, inverse_flattening=299.3249646)
    projection = iris.coord_systems.TransverseMercator(49.0, -2.0, 400000.0, -100000.0, 0.9996012717,
                                                       ellipsoid=spheroid)
    offshore_shape = shape.load_shp(offshore, keep_invalid=True, coord_system=projection)
    for sh in offshore_shape:
        if not sh.is_valid:
            sh.data = sh.data.buffer(0)
            sh.is_valid = sh.data.is_valid

    # Different natural earth required to get England, Scotland and Wales separately
    natural_earth2 = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_map_units')
    gb_shape = shape.load_shp(natural_earth2, name=['England', 'Scotland', 'Wales']).unary_union()
    trans_offshore = offshore_shape[0].transform_coord_system(gb_shape.coord_system)
    trans_offshore.data = trans_offshore.data.buffer(0)
    trans_offshore.is_valid = trans_offshore.data.is_valid
    final_shape_gb = gb_shape.union(trans_offshore)

    # list of European countries to consider
    european_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czech Republic', 'Denmark', 'Finland',
                          'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania',
                          'Luxembourg', 'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
                          'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland']

    # calculate country totals by dividing weighting in all grid cells by the sum of weightings
    total_weights = np.empty([29])
    # GB
    total_weights[0] = country_total_weights(final_shape_gb, cap_weights)
    # NI
    final_shape_ni = shps.filter(region='Northern Ireland').unary_union()
    total_weights[1] = country_total_weights(final_shape_ni, cap_weights)
    # Europe
    for i in range(len(european_countries)):
        final_shape = shps.filter(admin=european_countries[i]).unary_union()
        total_weights[i+2] = country_total_weights(final_shape, cap_weights)

    # national levels of installed capacity (change for different sensitivity scenarios)
    national_levels = np.array([28.7044, 114.4, 114.4,  # GB
                                1.4,     5.6,     5.6,    # NI
                                2.8781, 2.8781, 7.8996,  # Austria AT
                                4.2301, 4.2301, 11.6105,  # Belgium BE
                                0.6444, 0.6444, 1.7687,  # Bulgaria BG
                                0.9114, 0.9114, 2.5016,  # Croatia HR
                                0.3298, 0.3298, 0.9052,  # Czech rep CZ
                                6.7167, 6.7167, 18.4356,  # Denmark DK
                                2.4399, 2.4399, 6.6969,  # Finland FI
                                17.3442, 17.3442, 47.6054,  # France FR
                                63.4961, 63.4961, 174.2805,  # Germany DE
                                2.8575, 2.8575, 7.8431,  # Greece GR
                                0.3844, 0.3844, 1.0551,  # Hungary HU
                                4.0267, 4.0267, 11.0523,  # Ireland IE
                                10.8624, 10.8624, 29.8145,  # Italy IT
                                0.0531, 0.0531, 0.1457,  # Latvia LV
                                0.5356, 0.5356, 1.4701,  # Lithuania LT
                                0.1535, 0.1535, 0.4213,  # Lux LU
                                0.1175, 0.1175, 0.3225,  # Mont ME
                                6.119, 6.119, 16.7951,  # Netherlands NL
                                2.9549, 2.9549, 8.1104,  # Norway NO
                                5.9153, 5.9153, 16.2360,  # Poland PO
                                5.4686, 5.4686, 15.0099,  # Portugal PT
                                2.9813, 2.9813, 8.1829,  # Romania RO
                                0.0031, 0.0031, 0.0085,  # Slovakia SK
                                0.0032, 0.0032, 0.0088,  # Slovenia SL
                                24.1435, 24.1435, 66.2677,  # Spain ES
                                9.2226, 9.2226, 25.31376,  # Sweden SE
                                0.0868, 0.0868, 0.2382,  # Switzerland CH
                                ]).reshape(29, 3)

    in_dir = '/project/BG_Consultancy/NIC/data'

    # realisations
    realisations = list(np.arange(1, 41))
    bad_realisations = ['19', '22', '23', '25', '27', '29'] # missing data was infilled for these realisations

    for real in realisations:
        if real < 10:
            real = '0' + str(real)
        else:
            real = str(real)
        if real in bad_realisations:
            # load bias corrected / 100m estimated wind
            daily_wind_cube = iris.load_cube(f"{in_dir}/DePreSys_Bias_Correct/dep_gam_estimate_100m_wind/"
                                             f"missing_data_filled/"
                                             f"estimated_depresys_wind_100m_1959-2016_r0{real}_final_datafill.nc")
        else:
            daily_wind_cube = iris.load_cube(f"{in_dir}/DePreSys_Bias_Correct/dep_gam_estimate_100m_wind/"
                                             f"estimated_depresys_wind_100m_1959-2016_r0{real}_final.nc")
        period = '1959-2016'
        wind_dep_to_generation_csv(daily_wind_cube, region=region, shps=shps,
                                   turbine_option=arg_list[1], cap_scenario=arg_list[2],
                                   enercon_file=enercon_file, gamesa_file=gamesa_file, vestas_file=vestas_file,
                                   ng_onshore_file=ng_onshore_file, ng_offshore_file=ng_offshore_file,
                                   cap_weights=cap_weights, final_shape_gb=final_shape_gb,
                                   european_countries=european_countries, total_weights=total_weights,
                                   national_levels=national_levels, ens=real, period=period)


if __name__ == '__main__':
    main()
