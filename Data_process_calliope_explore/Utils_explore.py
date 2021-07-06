#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Read files
def read_efiles(type_event, n_event, n_var, n_type, dat):
    import xarray as xr
    import pandas as pd
    #type_event = "most_extreme_events"
    #n_event =['event1','event2','event3']
    #n_var = ['ssr','tas','windspeed']
    #dat = ['summer_wind_drought','winter_wind_drought','summer_surplus_generation']
    #n_type = ['duration', 'severity']
    maindir = '/badc/deposited2021/adverse_met_scenarios_electricity/data/'
    fnames_D = []
    data_D =[]
    for id_var in n_var:
        tmp = maindir + dat + '/uk/' + type_event +'/' + n_type + '/'+ n_event + '/' + dat + '_uk_' + type_event +'_' + n_type + '_' + n_event + '_' + id_var +'.nc'
        fnames_D.append(tmp)
        d_tmp = xr.open_dataset(tmp)
        data_D.append(d_tmp)
                
    return(data_D)


def simpleplot(dat, mvar, iday):
    
    dat = dat.sel(latitude = slice(33.33,73.89), longitude  = slice(-12.5, 34.17))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(NaturalEarthFeature('cultural', 'admin_0_countries', '10m'),
                       facecolor='none', edgecolor='black')
    ax.set_extent([-11, 31, 34, 71])
    # select same coordinates
    if (mvar == 't2m'):
        print(mvar)
        dat[mvar].values = dat[mvar].values - 273.5 
        dat[mvar].attrs["units"] = "deg C"
        dat[mvar][iday,:,:].plot(ax=ax, transform=ccrs.PlateCarree())
    
    dat[mvar][iday,:,:].plot(ax=ax, transform=ccrs.PlateCarree())
    

    
def sumdays_th(dat, mvar, probs, typex):
    
    dat = dat.sel(latitude = slice(33.33,73.89), longitude  = slice(-12.5, 34.17))
                  
    if (mvar!="wind_speed"):
        #probs=90
        #qt_dims = ("latitude", "longitude")
        qt_dims = 'time'
        if (typex == "high"):
            #qt_values = (probs)
            ds_qt = dat.quantile(probs, dim=qt_dims)
            # extract exceedances
            ex=xr.where(dat[mvar] > ds_qt, 1, 0)
        else:
            ds_qt = dat.quantile(probs, dim=qt_dims)
            # extract exceedances
            ex=xr.where(dat[mvar] < ds_qt, 1, 0)
        #ds_qt[mvar].plot()
    else:
        if (typex == "high"):
            ex=xr.where(dat[mvar] > probs, 1, 0)
            ex=ex.to_dataset()
            ex=ex.transpose('latitude','longitude','time')
        else:
            ex=xr.where(dat[mvar] < probs, 1, 0)
            ex=ex.to_dataset()
            ex=ex.transpose('latitude','longitude','time')
              
    lats =dat.latitude.values
    lon =dat.longitude.values
   
    #m_max =  np.zeros((len(lats),len(lon),1))
    m_max=0*ex[mvar][:,:,0]
    for ila in range(ex[mvar].shape[0]):
        for ilo in range(ex[mvar].shape[1]):
            a=ex[mvar][ila,ilo,:].values
            result = accumulate(a, lambda acc, elem: acc + elem if elem else 0)
            lr = list(result)
            m_max[ila,ilo]=max(lr)
            m_max.attrs["units"] = "days"
            #m_out=xr.DataArray(m_max)
    
    return[ex,m_max]




def vis_season(dat):
    main_plot = dat.plot(col='season', col_wrap=2, figsize=(10, 10),cmap='RdBu_r',
                                    transform=ccrs.PlateCarree(),
                                    subplot_kws={'projection': ccrs.PlateCarree()},
                        cbar_kwargs={"label": "Days"})
                            

    for ax in main_plot.axes.flat:
        ax.add_feature(NaturalEarthFeature('cultural', 'admin_0_countries', '10m'), facecolor='none', edgecolor='black')
        ax.set_extent([-11, 31, 34, 71])
        ax.set_ylabel('Days)',fontsize=14)
        
        
        
def compound_T2wind(tdat,wdat):
    # tdat: temperature exceedances over threshold
    # wdat: wind speend over threshol (in this case fixed)
    # note: need to regrid wind and temp to same grid
    lats =tdat.latitude.values
    lon =tdat.longitude.values
    #cmat =  np.zeros((len(lats),len(lon),365))
    cmat=0*tdat.t2m[:,:,:]
    for ila in range(len(lats)):
        for ilo in range(len(lon)):
            #print(ila)
            #print(ilo)
            temp=tdat['t2m'][ila,ilo,:].values & wdat['wind_speed'][ila,ilo,:].values
            cmat[ila,ilo,:]=temp
            #cmat= xr.DataArray(cmat,dims=["latitude", "longitude", "time"])
            
    return(cmat)



def plot_max_ex(dat):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(NaturalEarthFeature('cultural', 'admin_0_countries', '10m'),
                       facecolor='none', edgecolor='black')
    ax.set_extent([-11, 31, 34, 71])
    dat.plot(ax=ax, transform=ccrs.PlateCarree())

