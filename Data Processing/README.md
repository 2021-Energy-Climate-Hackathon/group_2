# Data Processing

The Met Office have produced a dataset of 'Adverse Weather Scenarios for Future
Electricity Systems'. This lets us explore and anticipate the challenges energy
systems may face under a warmer future climate.

This code was put together quickly during the Energy-Climate Hackathon 21-25
June 2021. It is not guaranteed to work as expected, and filepaths will need to
be altered to match data location on your system. Planned update in future to
improve portability, readability, reliability of this work, and to expand on
the analysis..

## Contents

`energy_functions_22.py` - useful functions for converting weather variables
into demand / windpower. Modified and simplified from Bloomfield et al.
https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/met.1858

`process_events.py` - Process reanalysis and DePreSys events data into daily
nationally aggregated data, and apply energy model conversion. Outputs `.nc`
files containing temperature, Cooling Degree Days (CDD), Heating Degree Days
(HDD), Demand, Wind speed, Windpower. 

`process_events_multi.py` - Imports functions from `process_events.py`, but
script is adapted for analysing in bulk events for different return periods.

`Winter Shortfall Plots.ipynb` - Using outputs of `process_events.py` and
`process_events_multi.py`, explore the weather and energy data. In this
example, the winter wind drought long duration events.

## Setup

### Data
Files in this directory assume working on the [jasmin](http://jasmin.ac.uk/)
server - filepaths in `data_paths.py` must be adjusted accordingly to point to
your data.

You can download the relevant data from CEDA archive, for example:

```bash
wget -e robots=off --mirror --no-parent -r https://dap.ceda.ac.uk/badc/deposited2021/adverse_met_scenarios_electricity/data/winter_wind_drought/uk/
```

Additional data corresponding to wind turbine power curves, demand model
regress coefficients, etc. should be placed under `wind_resources`.

Processed data will be saved to `data/` (this output path can be changed to any
other suitable filepath).

### Packages

Code tested on `python 3.7`

Essential: `[cartopy, numpy, pandas, shapely, xarray]` 

## Attribution

This work is derived from resources provided by organisers and participants of
the hackathon. Credits to Laura Dawkins and Hannah Bloomfield for providing
template code, and assistance with the data and models.  Contributions to this
work from [James Fallon](mailto:j.fallon@pgr.reading.ac.uk) and [Josh
Macholl](mailto:josh.macholl@hotmail.co.uk).

## License

Code produced during the energy-hackathon is, unless otherwise specified,
open-access, and aims to be reproducible. Software license as specified by
[2021-Energy-Climate-Hackathon](https://github.com/2021-Energy-Climate-Hackathon)
organisers.
