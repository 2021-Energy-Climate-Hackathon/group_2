"""
Convert estimated gridded daily solar radiation to regional average daily solar generation.
Firstly cuts out region, calculates solar capacity factor based on UoR solar model in each grid cell then
applies installed capacity weighting, sums this over the region and multiples by the national/regional level of
installed solar, to give daily regional average solar generation.

scitools default/current

Project: NIC 2020
DePreSys version of SolarGenModel.py
SolarGenModel.py was modified by Megan Pearce from Isabel Rushby and Laura Dawkins WindGenModel.py.

SPICE run time:
SS1 -  ~15 minutes (all realisations)
SS2 -  1hr 30 mins (all realisations)
"""

import os
import stat
import iris
from ascend import shape
import numpy as np
import pandas as pd
import argparse
import cartopy.io.shapereader as shpreader
import cf_units


def solrad_depresys_to_generation_csv(solrad_cube, temp_cube, shps, region, cap_scenario, cap_weights, final_shape_gb,
                                      european_countries, total_weights, national_levels, ens, period):
    """
    For specified ensemble member (ens) and timeframe (period):
    1.Calculate solar capacity factor (potential capacity) and combine with capacity weights.
    2.Sum weighted capacity factors over region (GB, Northern Ireland)
    3.Use chosen capacity scenario (Installed capacity scenario) to multiply with region weighted capacity factors to
    get daily solar generation cube. (Add GB and NI together to get solar generation for UK.)
    4.Convert data to a dataframe and save.
    If region Europe: repeat steps 2-4 for each country and add column to dataframe for each country.
    """
    # calculate solar capacity factor and combine with capacity grid weights
    cf = capacity_factor(temp_cube, solrad_cube)
    cf.data = cf.data * cap_weights.data

    # GB
    # mask capacity factor cube with GB shape, sum over lat and long
    solar_gen_cube_gb = final_shape_gb.mask_cube(cf).collapsed(['latitude', 'longitude'], iris.analysis.SUM)
    # Northern Ireland
    final_shape_ni = shps.filter(region='Northern Ireland').unary_union()
    # mask capacity factor cube with GB shape, sum over lat and long
    solar_gen_cube_ni = final_shape_ni.mask_cube(cf).collapsed(['latitude', 'longitude'], iris.analysis.SUM)

    # cap_scenario relates to the chosen input for the sensitivity study
    gb_factor = national_levels.iloc[0, cap_scenario - 1] / total_weights[0]
    solar_gen_cube_gb = iris.analysis.maths.multiply(solar_gen_cube_gb, gb_factor, dim=None, in_place=False)
    ni_factor = national_levels.iloc[1, cap_scenario - 1] / total_weights[1]
    solar_gen_cube_ni = iris.analysis.maths.multiply(solar_gen_cube_ni, ni_factor, dim=None, in_place=False)
    # Add GB and NI together
    solar_gen_cube_uk = solar_gen_cube_gb.data + solar_gen_cube_ni.data

    # convert cube to dataframe ready to write to csv file, indexed by dates
    # extract dates from cube
    dates = solrad_cube.coord('time').units.num2date(solrad_cube.coord('time').points)
    date_index = []
    for date in dates:
        date_index.append(date.strftime("%Y-%m-%d"))
    solar_gen_df = pd.DataFrame(solar_gen_cube_uk.data, index=date_index)
    solar_gen_df.columns = ['UK']

    os.makedirs(f"/data/users/mpearce/NIC/data/depresys/solar_generation/{ens}", exist_ok=True)
    # If region is UK, can save out solar_gen_df as is
    if region == 'United Kingdom':
        filename_csv = f"/data/users/mpearce/NIC/data/depresys/solar_generation/{ens}/SG_" \
                       + region.replace(" ", "_").lower() \
                       + '_cap_scenario=' + str(cap_scenario) \
                       + f"_test_{period}_biascorrected.csv"
        with open(filename_csv, 'w') as csv_file:  # write to csv
            solar_gen_df.to_csv(path_or_buf=csv_file)
        os.chmod(f"{filename_csv}",
                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IXOTH | stat.S_IROTH)  # change permissions
    # if region is Europe, need to do the same for each European country and sum together with UK
    if region == 'Europe':
        # loop over European countries
        for i in range(len(european_countries)):
            final_shape = shps.filter(admin=european_countries[i]).unary_union()
            solar_gen_cube = final_shape.mask_cube(cf).collapsed(['latitude', 'longitude'], iris.analysis.SUM)

            factor = national_levels.iloc[i + 2, cap_scenario - 1] / total_weights[i + 2]
            solar_gen = iris.analysis.maths.multiply(solar_gen_cube, factor, dim=None, in_place=False)
            # Add a column to solar_gen_df for the given country
            solar_gen_df[european_countries[i]] = pd.DataFrame(solar_gen.data, index=date_index)
        # Sum over countries
        solar_gen_df["Total"] = solar_gen_df.sum(axis=1)
        # Save out all countries
        filename_csv = f"/data/users/mpearce/NIC/data/depresys/solar_generation/{ens}/SG_" \
                       + region.replace(" ", "_").lower() \
                       + '_cap_scenario=' + str(cap_scenario) \
                       + f"_test_{period}_biascorrected.csv"
        with open(filename_csv, 'w') as csv_file:  # write to csv
            solar_gen_df.to_csv(path_or_buf=csv_file)
        os.chmod(f"{filename_csv}",
                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IXOTH | stat.S_IROTH)  # change permissions


def capacity_factor(temp, sol_rad):
    """
    Input - DePreSys 2m temp and estimated solar radiation as NetCDF file
    Purpose - Calculate solar power capacity factor each grid point.
    Output - gridded daily solar capacity factor to one NetCDF file.


    Method Notes:
    Calculate relative efficiency of panels at each grid point (pe_array)
        η(G,T)= ηr [1 − βr(Tc−Tr)]
        Where ηr is the photovoltaic cell efficiency evaluated at the reference temperature Tr
            ηr  is set to a constant value of 0.90
        βr is the fractional decrease of cell efficiency per unit temperature increase
         βr  is set to constant 0.00042 oC
        Tc is the cell temperature (assumed to be identical to the grid box temperature, i.e., T=Tc).
        η(G,T)= 0.90[1−0.00042(Tc−25)]

    Calculate solar power capacity factor each grid point (cf_array)
        CF(t)= power/powerSTC =η(G,T)(G(t))/(GSTC(t))
        Where G is the incoming surface solar radiation, T is the grid box 2m temperature and t is the time step (days).
        STC stands for standard test conditions (T=25℃, G=1000 Wm^(−2)) and η is the relative efficiency of the panel.

    """
    # Calculate relative efficiency of panels at each grid point (pe_array)
    pe_array = 0.90 * (1 - (0.00042 * (temp.data - 25)))
    # Calculate solar power capacity factor each grid point (cf_array)
    cf_array = pe_array * (sol_rad.data / 1000)
    # convert array to cube
    cf_cube = iris.cube.Cube(cf_array, long_name='solar_capacity_factor', attributes=None,
                             dim_coords_and_dims=[(sol_rad.coord('time'), 0),
                                                  (sol_rad.coord('latitude'), 1),
                                                  (sol_rad.coord('longitude'), 2)])
    return cf_cube


def country_total_weights(country_shape, weight_cube):
    """
    Divides weighting in all grid cells in a particular country by the sum of weightings in that country
    """
    masked_cube = country_shape.mask_cube(weight_cube)  # mask where is not the country being used
    country_sum = masked_cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM)  # sum weightings across country
    return country_sum.data


def main():
    """
    arg_list[0] - regions: UK or Europe (1 for UK, other for Europe)
    arg_list[1] - cap scenario to be simulated

    Inputs -
    in_dir: directory of data
    capacity weights: based on potential solar locations from James Price's research
    National levels: The total installed capacity of solar renewables in each country for each scenario we consider
    ens_mem: ensemble member number - DePreSys 001-040
    cubes: cubes of DePreSys mean daily 2m temperature and estimated solar radiation
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
    # set directory for loading saving files
    in_dir = '/project/BG_Consultancy/NIC/data'

    # load natural earth admin state provinces for filtering throughout script
    natural_earth = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    shps = shape.load_shp(natural_earth)

    # Weighting of UK potential solar locations from uk_locations_solrad.py
    cap_weights_file = '/project/BG_Consultancy/NIC/data/joined_weights_solrad_depresys.nc'
    cap_weights = iris.load_cube(cap_weights_file)
    weights = iris.analysis.cartography.area_weights(cap_weights, normalize=True)
    cap_weights.data = cap_weights.data * weights

    # different natural earth required to get England, Scotland, Wales separately
    natural_earth2 = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_map_units')
    gb_shape = shape.load_shp(natural_earth2, name=['England', 'Scotland', 'Wales']).unary_union()

    # list of European countries to consider
    european_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czech Republic', 'Denmark', 'Finland',
                          'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania',
                          'Luxembourg', 'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
                          'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland']

    # calculate country totals by dividing weighting in all grid cells by the sum of weightings
    total_weights = np.empty([29])
    # GB
    total_weights[0] = country_total_weights(gb_shape, cap_weights)
    # NI
    final_shape_ni = shps.filter(region='Northern Ireland').unary_union()
    total_weights[1] = country_total_weights(final_shape_ni, cap_weights)
    for i in range(len(european_countries)):
        final_shape = shps.filter(admin=european_countries[i]).unary_union()
        total_weights[i+2] = country_total_weights(final_shape, cap_weights)

    # national levels of installed capacity (change for different sensitivity scenarios)
    # load installed solar capacity csv to dataframe with country name as index
    # column 1 = current capacity (GW)) - cap scenario 1 and 2
    # column 2 = future capacity (GW) - cap scenario 3 and 4 - only this column used for DePreSys.
    # column 3 = future capacity (GW) * 1.5 - cap scenario 5 and 6
    csv_path = '/project/BG_Consultancy/NIC/data/solar_installed_capacity2.csv'
    national_levels = pd.read_csv(csv_path, index_col=0)

    # realisations
    realisations = list(np.arange(1, 41))
    bad_realisations = ['19', '22', '23', '25', '27', '29']  # missing data was infilled for these realisations

    for real in realisations:
        if real < 10:
            real = '0' + str(real)
        else:
            real = str(real)
        if real in bad_realisations:
            # load bias corrected / temp and estimated solar radiation
            solrad_cube = iris.load_cube(f"{in_dir}/DePreSys_Bias_Correct/DePreSys_solar/"
                                         f"bias_correct_estimated_DePreSys_solar_radiation_1959-2016_r0{real}_Gamma.nc")
            temp_cube = iris.load_cube(f"{in_dir}/DePreSys_Bias_Correct/DePreSys_Temp/"
                                       f"missing_data_filled/"
                                       f"bias_correct_depresys_temp_1959-2016_r0{real}_datafill.nc")
        else:
            solrad_cube = iris.load_cube(f"{in_dir}/DePreSys_Bias_Correct/DePreSys_solar/"
                                         f"bias_correct_estimated_DePreSys_solar_radiation_1959-2016_r0{real}_Gamma.nc")
            temp_cube = iris.load_cube(f"{in_dir}/DePreSys_Bias_Correct/DePreSys_Temp/"
                                       f"bias_correct_depresys_temp_1959-2016_r0{real}.nc")
            # fix temp cube from R processing
            temp_cube.coord('time').units = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        solrad_cube.coord('time').units = cf_units.Unit('days since 1970-01-01 00:00:00', calendar='gregorian')
        period = '1959-2016'
        solrad_depresys_to_generation_csv(solrad_cube, temp_cube, shps=shps, region=region,  cap_scenario=arg_list[1],
                                          cap_weights=cap_weights, final_shape_gb=gb_shape,
                                          european_countries=european_countries, total_weights=total_weights,
                                          national_levels=national_levels, ens=real, period=period)


if __name__ == '__main__':
    main()
