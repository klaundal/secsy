#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:57:56 2023

@author: aohma
"""

import numpy as np

import matplotlib.pyplot as plt

class CSplot(object):
    def __init__(self,ax,csgrid, **kwargs):
        
        # Add ax and grid to csax
        self.ax = ax
        self.grid = csgrid
        self.ax.set_aspect('equal')
        
        # set ax limits
        self.ax.set_xlim((self.grid.xi_min,self.grid.xi_max))
        self.ax.set_ylim((self.grid.eta_min,self.grid.eta_max))
        
        # Set gridtype
        if 'gridtype' in kwargs.keys():
            gridtype = kwargs.pop('gridtype')
            if gridtype not in ['geo','dipole','apex','cs']:
                print("gridtype must be 'geo','dipole','apex' or 'cs' to be added.")
                gridtype=None
        else:
            gridtype = None
        
        # Longitude or local time
        if 'lt' in kwargs.keys():
            lt = bool(kwargs.pop('lt'))
        else:
            lt = False
        
        
        # Add grid
        if gridtype is not None:
            # Set default grid properties
            if 'linewidth' not in kwargs.keys():
                kwargs['linewidth'] = .5
    
            if 'color' not in kwargs.keys():
                kwargs['color'] = 'lightgrey'
            
            
            # Set grid resolution
            if 'lat_levels' in kwargs.keys():
                lat_levels = kwargs.pop('lat_levels')
            elif 'lat_res' in kwargs.keys():
                lat_res = kwargs.pop('lat_res')
                lat_levels = np.arange(-90,90,lat_res)[1:]
            else:
                lat_levels = np.arange(-90,90,10)[1:]
                
            if 'lon_levels' in kwargs.keys():
                lon_levels = kwargs.pop('lon_levels')
            elif 'lon_res' in kwargs.keys():
                lon_res = kwargs.pop('lon_levels')
                if lt:
                    lon_levels = np.arange(0,24,lon_res)
                else:
                    lon_levels = np.arange(0,360,lon_res)
            else:
                if lt:
                    lon_levels = np.r_[0:240:2]
                else:
                    lon_levels = np.r_[0:360:30]
            
            
            # Add the selected grid
            if gridtype =='cs':
                self.ax.set_xlabel('xi')
                self.ax.set_ylabel('eta')
                self.ax.grid(**kwargs)
            elif gridtype == 'km':
                self.add_km(res)
            else:
                self.add_grid(lat_levels=lat_levels,lon_levels=lon_levels,gridtype=gridtype,lt=lt,**kwargs)
            

        
        # Remove ticks and tickmarks
        if gridtype!='cs':
            self.ax.xaxis.set_tick_params(labelbottom=False)
            self.ax.yaxis.set_tick_params(labelleft=False)

            self.ax.set_xticks([])
            self.ax.set_yticks([])
     
    def add_grid(self,lat_levels=np.r_[-80:90:10],lon_levels=np.r_[0:360:30],gridtype='geo',lt=False,**kwargs):
        
        # Add latitudinal parallels
        lon=np.linspace(0,360,361) % 360
            
        if gridtype=='geo':
            xi,eta = self.grid.projection.geo2cube(*np.meshgrid(lon,lat_levels))
            
        self.ax.plot(xi.T,eta.T,**kwargs)
        
        # Add longitudinal meridians
        lat=np.linspace(-90,90,181)
        
        if lt:
            # Convert lon_levels to longitude
            pass
        

        
        xi,eta = self.grid.projection.geo2cube(*np.meshgrid(lon_levels,lat))
            
        self.ax.plot(xi,eta,**kwargs)
        
        
        
        # Add ticks
        iii = self.grid.ingrid(*np.meshgrid(lon,lat_levels)) # gridpoints in csgrid
        lon_mean = anglemean(np.where(~iii,np.nan,lon[None,:]),axis=1)
        lon_count = np.sum(iii,axis=1)
        lon_res=np.mean(np.diff(lon_levels))
        lon_pos = lon_mean//lon_res*lon_res + lon_res/2
        
        [self.text(x,y,str(int(y)), horizontalalignment='center',verticalalignment='center') for x,y in zip(lon_pos[lon_count>10],lat_levels[lon_count>10])]
        
        
        iii = self.grid.ingrid(*np.meshgrid(lon_levels,lat)) # gridpoints in csgrid
        lat_mean = np.nanmean(np.where(~iii,np.nan,lat[:,None]),axis=0)
        lat_count = np.sum(iii,axis=0)
        lat_res=np.mean(np.diff(lat_levels))
        lat_pos = lat_mean//lat_res*lat_res + lat_res/2
        [self.text(x,y,str(int(x)), horizontalalignment='center',verticalalignment='center') for x,y in zip(lon_levels[lat_count>10], lat_pos[lat_count>10])]
        
        pass
    
    def add_km(self,resolution):
        
        pass
    
    def text(self, lon, lat, text, ignore_limits=False, **kwargs):
        """
        Wrapper for matplotlib's text function. Accepts lat, lt instead of x and y
        keywords passed to this function is passed on to matplotlib's text
        """

        xi,eta = self.grid.projection.geo2cube(lon,lat)

        if self.grid.ingrid(lon,lat):
            return self.ax.text(xi, eta, text, **kwargs)
        else:
            print('text outside plot limit - set "ignore_limits = True" to override')
    
     
    def plotgrid(self,dd):
        pass

    def plotgrid2(self, labels=False, **kwargs):
        """ plot lt, lat-grid on self.ax
        parameters
        ----------
        labels: bool
            set to True to include lat/lt labels
        **kwarsgs: dictionary
            passed to matplotlib's plot function (for linestyle etc.)
        """
        returns= []
        for lt in [0, 6, 12, 18]:
            returns.append(self.plot([self.minlat, 90], [lt, lt], **kwargs))

        lts = np.linspace(0, 24, 100)
        for lat in np.r_[90: self.minlat -1e-12 :-10]:
            returns.append(self.plot(np.full(100, lat), lts, **kwargs))

        # add LAT and LT labels to axis
        if labels:
            returns.append(self.writeLATlabels())
            returns.append(self.writeLTlabels())
        return tuple(returns)
        
    def plot(self,lon,lat,**kwargs):
        x,y = self.grid.projection.geo2cube(lon,lat)
        return self.ax.plot(x,y,**kwargs)
     
    def contourf(self,*args,**kwargs):
        
        if len(args)==1: # Only C provided
            self.ax.contourf(self.grid.xi_mesh,self.grid.eta_mesh,args[0],**kwargs)
        elif len(args)==3:
            X,Y = self.grid.projection.geo2cube(args[0],args[1])
            self.ax.contourf(X,Y,args[2],**kwargs)
        else:
            raise TypeError    
     
    def pcolormesh(self,*args,**kwargs):
        
        if len(args)==1: # Only C provided
            self.ax.pcolormesh(self.grid.xi_mesh,self.grid.eta_mesh,args[0],**kwargs)
        elif len(args)==3:
            X,Y = self.grid.projection.geo2cube(args[0],args[1])
            self.ax.pcolormesh(X,Y,args[2],**kwargs)
        else:
            raise TypeError
        
        
    def add_coastlines(self, resolution='110m' ,**kwargs):
         
        if 'color' not in kwargs.keys():
             kwargs['color'] = 'black'
 
    
        for cl in self.grid.projection.get_projected_coastlines(resolution = resolution):
            xi, eta = cl
            self.ax.plot(xi, eta, **kwargs)
            

def anglemean(X,axis=None):
    return np.rad2deg(np.arctan2(np.nanmean(np.sin(np.deg2rad(X)),axis=axis),np.nanmean(np.cos(np.deg2rad(X)),axis=axis)))
    