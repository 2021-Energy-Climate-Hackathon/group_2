# Simple code to load in a plots some of the Adverse Weather Scenarios for Future Electricity Systems data

# Load in R libraries (libraries may need to be installed first)
# install.packages('RNetCDF')
# install.packages('fields')
library(RNetCDF)
library(fields)

# Data repository on CEDA

dataset_loc = '/badc/deposited2021/adverse_met_scenarios_electricity/data/'

# Open one relevant netcdf file - e.g. UK winter wind drought, one of the most extreme events in terms of duration:

# Location of the event of interest
data_loc = paste0(dataset_loc,'winter_wind_drought/uk/most_extreme_events/duration/event1/')

# Open the temperature data file
nc = open.nc(paste0(data_loc,'winter_wind_drought_uk_most_extreme_events_duration_event1_tas.nc'))

# Look at the netcdf meta data
print.nc(nc)
# netcdf classic {
#   dimensions:
#     longitude = 85 ;
#     latitude = 81 ;
#     time = 730 ;
#     variables:
#       NC_DOUBLE longitude(longitude) ;
#     NC_CHAR longitude:units = "degrees_east" ;
#     NC_CHAR longitude:long_name = "longitude" ;
#     NC_DOUBLE latitude(latitude) ;
#     NC_CHAR latitude:units = "degrees_north" ;
#     NC_CHAR latitude:long_name = "latitude" ;
#     NC_DOUBLE time(time) ;
#     NC_CHAR time:units = "hours since 1970-01-01 00:00:00" ;
#     NC_CHAR time:long_name = "time" ;
#     NC_DOUBLE t2m(longitude, latitude, time) ;
#     NC_CHAR t2m:units = "K" ;
#     NC_DOUBLE t2m:_FillValue = NaN ;
#     
#     // global attributes:
#       NC_CHAR :Project = "Adverse weather scenarios for electricity systems – Met Office, National Infrastructure Commition and Climate Change Committee" ;
#       NC_CHAR :Event specification = "winter_wind_drought_uk_most_extreme_events_duration" ;
#       NC_CHAR :Event start date = "2006-12-01" ;
#       NC_CHAR :Event duration = "30 days" ;
#       NC_CHAR :Event severity = "19.307 (no units)" ;
#       NC_CHAR :Domain = "Europe" ;
#       NC_CHAR :Resolution = "60km" ;
#       NC_CHAR :Frequency = "daily" ;
#       NC_CHAR :Calendar = "gregorian" ;
#       NC_CHAR :Meteorological variable = "Surface air temperature" ;
#       NC_CHAR :Originating data source = "DePreSys" ;
# }

# Extract the longitude, latitude, time and meteorological inforamtion

lon = var.get.nc(nc,'longitude')
lat = var.get.nc(nc,'latitude')
time = var.get.nc(nc,'time')
# Turn time into a date (time is hours since 1970-01-01 00:00:00)
date = as.Date(time/24, origin = '1970-01-01')
temperature = var.get.nc(nc,'t2m')
close.nc(nc)

# Spatial plot of the first day of data
quilt.plot(expand.grid(lon,lat),temperature[,,1],xlab='Longitude',ylab-'Latitude')

# Time series plot in one grid cell
plot(date,temperature[1,1,])


# Similar for wind speed:

nc = open.nc(paste0(data_loc,'winter_wind_drought_uk_most_extreme_events_duration_event1_windspeed.nc'))
print.nc(nc)
lon = var.get.nc(nc,'longitude')
lat = var.get.nc(nc,'latitude')
time = var.get.nc(nc,'time')
date = as.Date(time/24, origin = '1970-01-01')
windspeed = var.get.nc(nc,'windspeed')
close.nc(nc)

# Spatial plot of the first day of data
quilt.plot(expand.grid(lon,lat),windspeed[,,1],xlab='Longitude',ylab-'Latitude')

# Time series plot in one grid cell
plot(date,windspeed[1,1,])


# Similar for solar radiation:

nc = open.nc(paste0(data_loc,'winter_wind_drought_uk_most_extreme_events_duration_event1_ssr.nc'))
print.nc(nc)
lon = var.get.nc(nc,'longitude')
lat = var.get.nc(nc,'latitude')
time = var.get.nc(nc,'time')
date = as.Date(time/24, origin = '1970-01-01')
solar = var.get.nc(nc,'ssr')
close.nc(nc)

# Spatial plot of the first day of data
quilt.plot(expand.grid(lon,lat),solar[,,1],xlab='Longitude',ylab-'Latitude')

# Time series plot in one grid cell
plot(date,solar[1,1,])
