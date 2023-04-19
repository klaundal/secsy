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
            if gridtype not in ['geo','dipole','apex','km','cs']:
                print("gridtype must be 'geo','dipole','apex', 'km' or 'cs' to be added.")
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
            
            if 'km_levels' in kwargs.keys():
                km_levels = kwargs.pop('km_levels')
            else:
                km_levels = np.round(self.grid.R*(self.grid.xi_max+self.grid.eta_max)//5,-2)
            
            # Add the selected grid
            if gridtype =='cs':
                self.ax.set_xlabel('$\\xi$')
                self.ax.set_ylabel('$\\eta$')
                self.ax.grid(**kwargs)
            elif gridtype == 'km':
                self.add_km(km_levels,**kwargs)
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
        
        count_min = self.grid.R*(self.grid.xi_max+self.grid.eta_max)//300
        
        # Add ticks
        iii = self.grid.ingrid(*np.meshgrid(lon,lat_levels)) # gridpoints in csgrid
        lon_mean = anglemean(np.where(~iii,np.nan,lon[None,:]),axis=1)
        lon_count = np.sum(iii,axis=1)
        lon_res=np.mean(np.diff(lon_levels))
        lon_pos = lon_mean//lon_res*lon_res + lon_res/2
        
        [self.text(x,y,str(int(y)), horizontalalignment='center',verticalalignment='center') for x,y in zip(lon_pos[lon_count>count_min],lat_levels[lon_count>count_min])]
        
        
        iii = self.grid.ingrid(*np.meshgrid(lon_levels,lat)) # gridpoints in csgrid
        lat_mean = np.nanmean(np.where(~iii,np.nan,lat[:,None]),axis=0)
        lat_count = np.sum(iii,axis=0)
        lat_res=np.mean(np.diff(lat_levels))
        lat_pos = lat_mean//lat_res*lat_res + lat_res/2
        [self.text(x,y,str(int(x)), horizontalalignment='center',verticalalignment='center') for x,y in zip(lon_levels[lat_count>count_min], lat_pos[lat_count>count_min])]
        
        pass
    
    def add_km(self,resolution,**kwargs):
        
        
        csres=0.005
        
        # xi gridlines
        eta = np.arange(self.grid.eta_min*2,self.grid.eta_max*2,csres)
        # xi(eta) for xi>0
        xi = np.arange(0,self.grid.xi_max*1.1,csres)
        
        diff = self.grid.projection.differentials(*np.meshgrid(xi,eta),csres,0,R=self.grid.R)[0]
        diff[:,0]=0
        
        xi_pos = []
        for i in range(len(eta)): xi_pos.append(np.interp(np.arange(0,self.grid.L/2,resolution),np.cumsum(diff[i,:]),xi))
        xi_pos = np.array(xi_pos)  
        
        # xi(eta) for xi<0
        xi = np.arange(0,self.grid.xi_min*1.1,-csres)

        diff = self.grid.projection.differentials(*np.meshgrid(xi,eta),-csres,0,R=self.grid.R)[0]
        diff[:,0]=0
        
        xi_neg = []
        for i in range(len(eta)): xi_neg.append(np.interp(np.arange(-resolution,-self.grid.L/2,-resolution)[::-1],np.cumsum(diff[i,:])[::-1],xi[::-1]))
        xi_neg = np.array(xi_neg)
        
        self.ax.plot(np.hstack((xi_neg,xi_pos)),eta,**kwargs)
        
        # tickmarks
        idx = (np.abs(eta - self.grid.eta_min)).argmin()
        xi_tick = np.hstack((xi_neg,xi_pos))[idx,:]
        eta_tick = self.grid.eta_min*1.05
        km_tick = np.concatenate((np.arange(-resolution,-self.grid.L/2,-resolution)[::-1],np.arange(0,self.grid.L/2,resolution)))
        
        ind = (xi_tick>=self.grid.xi_min)&(xi_tick<=self.grid.xi_max)
        xi_tick = xi_tick[ind]
        km_tick = km_tick[ind]
        [self.ax.text(xi_tick[i],eta_tick,str(int(km_tick[i])),horizontalalignment='center',verticalalignment='top') for i in range(len(xi_tick))]
        
        
        
        ## eta gridlines
        
        xi = np.arange(self.grid.eta_min*2,self.grid.eta_max*2,csres)
        # eta(xi) for eta>0
        eta = np.arange(0,self.grid.eta_max*1.1,csres)
        
        diff = self.grid.projection.differentials(*np.meshgrid(xi,eta),0,csres,R=self.grid.R)[1].T
        diff[:,0]=0
        
        eta_pos = []
        for i in range(len(xi)): eta_pos.append(np.interp(np.arange(0,self.grid.W/2,resolution),np.cumsum(diff[i,:]),eta))
        eta_pos = np.array(eta_pos)  
        
        # xi(eta) for xi<0
        eta = np.arange(0,self.grid.eta_min*1.1,-csres)

        diff = self.grid.projection.differentials(*np.meshgrid(xi,eta),0,-csres,R=self.grid.R)[1].T
        diff[:,0]=0
        
        eta_neg = []
        for i in range(len(xi)): eta_neg.append(np.interp(np.arange(-resolution,-self.grid.W/2,-resolution)[::-1],np.cumsum(diff[i,:])[::-1],eta[::-1]))
        eta_neg = np.array(eta_neg)
        
        
        self.ax.plot(xi,np.hstack((eta_neg,eta_pos)),**kwargs)
        
                
        # tickmarks
        xi_tick = self.grid.xi_min*1.05
        idx = (np.abs(xi - self.grid.xi_min)).argmin()
        eta_tick = np.hstack((eta_neg,eta_pos))[idx,:]
        km_tick = np.concatenate((np.arange(-resolution,-self.grid.W/2,-resolution)[::-1],np.arange(0,self.grid.W/2,resolution)))
        
        ind = (eta_tick>=self.grid.eta_min)&(eta_tick<=self.grid.eta_max)
        eta_tick = eta_tick[ind]
        km_tick = km_tick[ind]
        [self.ax.text(xi_tick,eta_tick[i],str(int(km_tick[i])),horizontalalignment='right',verticalalignment='center') for i in range(len(eta_tick))]
        
   
        
    
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
            self.ax.contourf(self.grid.xi,self.grid.eta,args[0],**kwargs)
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
    