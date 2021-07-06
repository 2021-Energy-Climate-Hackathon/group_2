# README

This directory contains 3 python scripts that were previously developed for characteristing and identifying adverse weather scenarios for the 'Adverse Weather Scenarios for Future Electricity Systems' dataset. This code converts meteorological data to regional demand and generation using methods similar to Bloomfield et al. (2020):

1. DePreSys_DemandModel.py - Convert gridded daily 2m temperature to regional average daily demand. Cuts out region, takes averages temperature in this region and applies a chosen demand regression model to calculate daily weather dependent demand.

2. DePreSys_SolarGenModel.py - Convert gridded daily solar radiation and 2m temperature to regional daily solar generation 

3. DePreSys_WindGenModel.py - Convert gridded 100m wind speed to wind generation in regions 


