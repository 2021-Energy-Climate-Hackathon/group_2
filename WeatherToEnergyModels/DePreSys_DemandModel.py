"""
Convert gridded daily 2m temperature to regional average daily demand. Firstly cuts out region, then averages
temperature in this region, then applies a chosen demand regression model.

module load scitools/default-current

Project: NIC 2020
DePreSys version of DemandModel.py
Modified by Megan Pearce from Isabel Rushby and Laura Dawkins DemandModel.py.

SPICE run time:
SS1 - ~10 mins all realisations
SS2 - 3 hours all realisations
"""
import os
import stat
import iris
from ascend import shape
import cartopy.io.shapereader as shpreader
from iris import pandas
import numpy as np
import pandas as pd
import argparse
import cf_units


def temp_depresys_to_demand_csv(daily_temp_cube, demand_model, region, uk_demand_model_coeff, european_countries,
                                europe_demand_model_coeff, ens):
    """
    Average the daily mean temperature across a region, apply demand model and write to csv file
    daily_temp_cubes: cube of daily average 2m temperatures
    region: United Kingdom or Europe
    demand_model: which of the 3 demand models to use (UK, UK/France, France)
    """
    # Load shape files
    natural_earth = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    shps = shape.load_shp(natural_earth)
    # Firstly calculate national daily demand for UK
    final_shape = shps.filter(admin='United Kingdom').unary_union()
    # calculate the area of each grid cell for doing weighted area average
    if daily_temp_cube.coord('latitude').bounds is None:
        daily_temp_cube.coord('latitude').guess_bounds()
    if daily_temp_cube.coord('longitude').bounds is None:
        daily_temp_cube.coord('longitude').guess_bounds()
    weights = iris.analysis.cartography.area_weights(daily_temp_cube, normalize=True)
    # find daily area weighted average 2m temp in UK
    temp_cube = final_shape.mask_cube(daily_temp_cube).collapsed(['latitude', 'longitude'],
                                                                 iris.analysis.MEAN, weights=weights)
    # convert cube to dataframe ready to write to csv file
    temp_df = iris.pandas.as_data_frame(temp_cube, copy=True)
    # apply demand model based on regression models from UoR data set - to give daily UK demand
    demand_df = calc_demand(temp_df, coefs=uk_demand_model_coeff[demand_model - 1, :])

    os.makedirs(f"/data/users/mpearce/NIC/data/depresys/demand/", exist_ok=True)
    # If only interested in UK, save this daily demand data frame as csv
    if region == 'United Kingdom':
        filename_csv = f"/data/users/mpearce/NIC/data/depresys/demand/WDD_" \
                       + region.replace(" ", "_").lower() \
                       + '_demand_model=' + str(demand_model) \
                       + f"_r0{ens}.csv"
        # save out demand
        with open(filename_csv, 'w') as csv_file:  # write to csv
            demand_df.to_csv(path_or_buf=csv_file, header=['weather dep demand'])
        os.chmod(f"{filename_csv}",
                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IXOTH | stat.S_IROTH)  # change permissions

    # If interested in all of Europe, need to apply the demand models of each European country one-by-one and sum the
    # daily demand across Europe. Do this by added rows to the demand_df data frame for each country
    if region == 'Europe':
        # label the existing column as UK
        demand_df.columns = ['UK']
        # Loop over European countries
        for i in range(len(european_countries)):
            final_shape = shps.filter(admin=european_countries[i]).unary_union()
            # find daily area weighted average 2m temp in that country
            temp_cube = final_shape.mask_cube(daily_temp_cube).collapsed(['latitude', 'longitude'], iris.analysis.MEAN,
                                                                         weights=weights)
            # convert cube to dataframe ready to write to csv file
            temp_df = iris.pandas.as_data_frame(temp_cube, copy=True)
            # add column to the demand_df data frame for the new country
            demand_df[european_countries[i]] = calc_demand(temp_df, coefs=europe_demand_model_coeff[i, :])
        demand_df["Total"] = demand_df.sum(axis=1)
        # We only need the overall European total demand
        # demand_df = demand_df["Total"]
        filename_csv = f"/data/users/mpearce/NIC/data/depresys/demand/WDD_" \
                       + region.replace(" ", "_").lower() \
                       + '_demand_model=' + str(demand_model) \
                       + f"_r0{ens}.csv"
        with open(filename_csv, 'w') as csv_file:  # write to csv
            demand_df.to_csv(path_or_buf=csv_file)
        os.chmod(f"{filename_csv}",
                 stat.S_IRWXU | stat.S_IRWXG | stat.S_IXOTH | stat.S_IROTH)  # change permissions


def calc_demand(temp_ts, coefs):
    """
    Calculates weather dependent demand based on daily average temp
    Uses the approach of UoR - calculate heating degree days (hdd) and cooling degree days (cdd)
    and apply regression model using coefficients estimated by UoR.
    """
    df = temp_ts
    df.columns = ['temp']
    df.loc[df['temp'] < 15.5, 'hdd'] = 15.5-df['temp']
    df.loc[df['temp'] >= 15.5, 'hdd'] = 0
    df.loc[df['temp'] > 22, 'cdd'] = df['temp'] - 22
    df.loc[df['temp'] <= 22, 'cdd'] = 0
    df['wdd'] = coefs[0] + (coefs[1]*df['hdd']) + (coefs[2]*df['cdd'])
    df_ret = df['wdd']
    df_ret = df_ret.to_frame()
    return df_ret


def main():
    """
    files: directory and filename(s) of depresys temperature mean daily NetCDF files
    regions: regions to calculate mean temperatures and demand for (UK or Europe)
    european countries: list of countries
    demand_model_coeff: Apply the demand model of each country in Europe individually, define demand model
      regression coefficients based on Hannah Bloomfield's paper.

    arg_list[0] - regions: UK or Europe (1 for UK, other for Europe)
    arg_list[1] - demand model (1 = UK, 2 = UK + France HDD, 3 = UK heating and French cooling)
    arg_list[2] - period of choice (0 for historical - 1979-2018, 1 for future - 2019-2098, other for all - 1979-2098)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', help='delimited list input', type=str)
    args = parser.parse_args()
    arg_list = [int(item) for item in args.list.split(',')]

    if arg_list[0] == 1:
        region = 'United Kingdom'
    else:
        region = 'Europe'

    european_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czech Republic', 'Denmark', 'Finland',
                          'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania',
                          'Luxembourg', 'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
                          'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland']

    # Demand model regression coefficients from University of Reading documentation
    europe_demand_model_coeff = np.array([6.8, 0.09, 0.85,  # Austria AT
                                          9.6, 0.12, 0.11,  # Belgium BE
                                          3.8, 0.1, 0.13,  # Bulgaria BG
                                          1.9, 0.02, 0.09,  # Croatia HR
                                          7.1, 0.1, 0.11,  # Czech rep CZ
                                          3.5, 0.05, 0,  # Denmark DK
                                          7.9, 0.15, 0,  # Finland FI
                                          46.2, 2.02, 1.19,  # France FR
                                          56.4, 0.33, 1.23,  # Germany DE
                                          5.3, 0.11, 0.46,  # Greece GR
                                          4.8, 0.03, 0.09,  # Hungary HU
                                          3, 0.04, 0,  # Ireland IE
                                          33.7, 0.2, 2.77,  # Italy IT
                                          0.8, 0.01, 0,  # Latvia LV
                                          1.2, 0.01, 0,  # Lithuania LT
                                          0.5, 0, 0,  # Lux LU
                                          0.33, 0, 0.05,  # Mont ME
                                          13, 0.11, 0.37,  # Netherlands NL
                                          8.7, 0.46, 0,  # Norway NO
                                          18.8, 0.14, 0.3,  # Poland PO
                                          5.6, 0.1, 0.1,  # Portugal PT
                                          6.4, 0.07, 0.2,  # Romania RO
                                          3.2, 0.03, 0.06,  # Slovakia SK
                                          1.5, 0.01, 0.04,  # Slovenia SL
                                          28.5, 0.25, 0.93,  # Spain ES
                                          12, 0.36, 0,  # Sweden SE
                                          6.6, 0.06, 0  # Switzerland CH
                                          ]).reshape(27, 3)

    # The values in the above array are rounded to the nearest 0.01, which has a small impact on the output
    # (noticed when comparing our output to the Uni of Reading's...
    # so replace second and third columns with non-rounded values in Uni of Reading's supplementary file:
    coeffs = pd.read_csv("/project/BG_Consultancy/NIC/data/ERA5_reanalysis_models/demand_model_outputs/"
                         "ERA5_Regression_coeffs_demand_model.csv")
    hdd = np.array(coeffs.iloc[[8]])
    cdd = np.array(coeffs.iloc[[9]])
    europe_demand_model_coeff[:, 1] = hdd[0][1:28]
    europe_demand_model_coeff[:, 2] = cdd[0][1:28]
    # europe_demand_model_coeff values in columns 2 and 3 are now not rounded.

    # demand model 1 = UK, 2 = UK + France hdd, 3 = France cdd
    uk_demand_model_coeff = np.array([35.1, 0.747608687595952, 0, 35.1, 2.02472352801677, 0, 35.1,
                                      0.747608687595952, 1.19111495243485]).reshape(3, 3)

    # realisations
    realisations = list(np.arange(1, 41))
    bad_realisations = ['19', '22', '23', '25', '27', '29']

    for real in realisations:
        if real < 10:
            real = '0' + str(real)
        else:
            real = str(real)
        if real in bad_realisations:
            daily_temp_cube = iris.load_cube(f"/project/BG_Consultancy/NIC/data/DePreSys_Bias_Correct/DePreSys_Temp/"
                                             f"missing_data_filled/"
                                             f"bias_correct_depresys_temp_1959-2016_r0{real}_datafill.nc")
        else:
            daily_temp_cube = iris.load_cube(f"/project/BG_Consultancy/NIC/data/DePreSys_Bias_Correct/DePreSys_Temp/"
                                             f"bias_correct_depresys_temp_1959-2016_r0{real}.nc")
            new_time = cf_units.Unit(cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='gregorian'))
            daily_temp_cube.coord('time').units = new_time
            daily_temp_cube.convert_units('celsius')

        temp_depresys_to_demand_csv(daily_temp_cube, demand_model=arg_list[1], region=region,
                                    uk_demand_model_coeff=uk_demand_model_coeff,
                                    europe_demand_model_coeff=europe_demand_model_coeff,
                                    european_countries=european_countries, ens=real)


if __name__ == '__main__':
    main()
