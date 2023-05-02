#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:57:56 2023

@author: aohma
"""

import numpy as np


class CSplot(object):
    def __init__(self,ax,csgrid, **kwargs):
        '''
        A class creating a cubed sphere axis object to plot data with (lon,lat) corrdinates on a cubed sphere projection.
        
        
        Example:
        --------
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        csax = CSplot(ax,grid)
        csax.MEMBERFUNCTION()
        plt.show()
        
        where memberfunctions include:
        plot(lon,lat,**kwargs)            - works like plt.plot
        text(lon,lat,text, **kwargs)      - works like plt.text
        scatter(lon,lat,**kwargs)         - works like plt.scatter
        contour(lon,lat,c,**kwargs)                - works like plt.contour
        contourf(lon,lat,c,**kwargs)               - works like plt.contourf

        Parameters
        ----------
        ax : matplotlib.AxesSubplot
            A standard matplotlib AxesSubplot object.
        csgrid : secsy.cubedsphere.CSgrid
            A cubed sphere grid object.
        **kwargs : dict, optional
            Keywords to control grid lines.
            In addition to Line2D properties, the following can be specified:
            gridtype : str or None
                Determines which grid lines that are added to the csplot
                'geo' adds a geographic lon,lat grid
                'dipole' adds a magnetic dipole lon,lat grid
                'apex' adds a magnetic apex lon lat grid
                'km' adds gridlines with equal physical distance in km
                'cs' adds a cubed sphere xi,eta grid
                Default is None (no grid)
            lt : bool
                If lt is True, lon is replaced with local time.
                Default is False. 
            lat_levels : array_like
                Where to plot latitudinal grid parallels in spherical grids. If not provided, default values are used.
            lat_res : int
                Resolution of latitudinal grid parallels in spherical grids. Ignored if lat_levels are set.
            lon_levels : array_like
                Where to plot longitudinal grid meridians in spherical grids. If not provided, default values are used.
            lon_res : int 
                Resolution of longitudinal grid meridians in spherical grids. Ignored if lon_levels are set.
            km_res : int
                Resolution of 'km' grid. If not provided, default values are used.

        Returns
        -------
        None.

        '''
        
        # Add ax and grid to csax
        self.ax = ax
        self.grid = csgrid
        self.ax.set_aspect('equal')
        
        # set ax limits
        self.ax.set_xlim((self.grid.xi_min,self.grid.xi_max))
        self.ax.set_ylim((self.grid.eta_min,self.grid.eta_max))
        
        # Select gridtype
        if 'gridtype' in kwargs.keys():
            gridtype = kwargs.pop('gridtype')
            if gridtype not in ['geo','km','cs']:
                print("gridtype must be 'geo', 'km' or 'cs' to be added. 'dipole' and 'apex' will soon be available." )
                gridtype=None
        else:
            gridtype = None
        
        # Longitude or local time
        if 'lt' in kwargs.keys():
            lt = bool(kwargs.pop('lt'))
            print("'lt' not implemented yet.")
        else:
            lt = False
        
        
        # Add grid
        if gridtype is not None:
            # Set default grid linewidth
            if 'linewidth' not in kwargs.keys():
                kwargs['linewidth'] = .5
    
            # Set default grid color
            if 'color' not in kwargs.keys():
                kwargs['color'] = 'lightgrey'
            
            
            # Set latitudinal grid resolution
            if 'lat_levels' in kwargs.keys():
                lat_levels = kwargs.pop('lat_levels')
            elif 'lat_res' in kwargs.keys():
                lat_res = kwargs.pop('lat_res')
                lat_levels = np.arange(-90,90,lat_res)[1:]
            else: # Default resolution is 10 degrees
                lat_levels = np.arange(-90,90,10)[1:] 
                
            # Set longitidinal grid resolution
            if 'lon_levels' in kwargs.keys():
                lon_levels = kwargs.pop('lon_levels')
            elif 'lon_res' in kwargs.keys():
                lon_res = kwargs.pop('lon_levels')
                if lt:
                    lon_levels = np.arange(0,24,lon_res)
                else:
                    lon_levels = np.arange(0,360,lon_res)
            else: # Default res is 30 degrees / 2 hours
                if lt:
                    lon_levels = np.r_[0:240:2]
                else:
                    lon_levels = np.r_[0:360:30]
            
            # Set grid resolution in 'km' grid
            if 'km_levels' in kwargs.keys():
                km_levels = kwargs.pop('km_levels')
            else: # Default res depends on cs grid size
                km_levels = np.round(self.grid.R*(self.grid.xi_max+self.grid.eta_max)//5,-2)
            
            # Add the selected grid
            if gridtype =='cs':
                self.ax.set_xlabel('$\\xi$')
                self.ax.set_ylabel('$\\eta$')
                self.ax.grid(**kwargs)
            elif gridtype == 'km':
                self.add_km_grid(km_levels,**kwargs)
            else:
                self.add_spherical_grid(lat_levels=lat_levels,lon_levels=lon_levels,gridtype=gridtype,lt=lt,**kwargs)
            

        
        # Remove ticks and tickmarks
        if gridtype!='cs':
            self.ax.xaxis.set_tick_params(labelbottom=False)
            self.ax.yaxis.set_tick_params(labelleft=False)

            self.ax.set_xticks([])
            self.ax.set_yticks([])
     
    def add_spherical_grid(self,lat_levels=np.r_[-80:90:10],lon_levels=np.r_[0:360:30],gridtype='geo',lt=False,**kwargs):
        '''
        Adds a spherical lon/lat grid to the axis

        Parameters
        ----------
        lat_levels : array_like, optional
            Array with location of latitudinal parallels. The default is np.r_[-80:90:10].
        lon_levels : array_like, optional
            Array with location of longitudinal meridians. The default is np.r_[0:360:30].
        gridtype : str, optional
            Which coordinate system to add. The default is 'geo'.
        lt : bool, optional
            If lt is True, lon is replaced with local time. The default is False.
        **kwargs : dict
            Line2D properties.

        Returns
        -------
        None.

        '''
        
        ## Latitudinal parallels
    
        lon=np.linspace(0,360,361) % 360 # Longitidunal locations
        
        # Convert to cs coordinates
        if gridtype=='geo':
            xi,eta = self.grid.projection.geo2cube(*np.meshgrid(lon,lat_levels))
        
        # Plot the grid lines
        self.ax.plot(xi.T,eta.T,**kwargs)
        
        
        ## Longitudinal meridians
        lat=np.linspace(-90,90,181)# Latitudinal locations
        
        
        if lt: # Convert lon_levels from lt to longitude
            pass
        
        # Convert to cs coordinates
        if gridtype=='geo':
            xi,eta = self.grid.projection.geo2cube(*np.meshgrid(lon_levels,lat))

        # Plot the grid            
        self.ax.plot(xi,eta,**kwargs)
        
        # Minimum "length" of grid line from tick t be plotted
        count_min = self.grid.R*(self.grid.xi_max+self.grid.eta_max)//300
        
        # Add latitudinal ticks
        iii = self.grid.ingrid(*np.meshgrid(lon,lat_levels)) # points in csgrid
        lon_mean = anglemean(np.where(~iii,np.nan,lon[None,:]),axis=1) # mean of lon grid lines
        lon_count = np.sum(iii,axis=1) # "length" of grid lines
        lon_res=np.mean(np.diff(lon_levels)) # Distance between meridians
        lon_pos = lon_mean//lon_res*lon_res + lon_res/2 # Move tick location from mean to between meridians
        
        # Add the latitudinal ticks
        [self.text(x,y,str(int(y)), horizontalalignment='center',verticalalignment='center') for x,y in zip(lon_pos[lon_count>count_min],lat_levels[lon_count>count_min])]
        
        
        iii = self.grid.ingrid(*np.meshgrid(lon_levels,lat)) # points in csgrid
        lat_mean = np.nanmean(np.where(~iii,np.nan,lat[:,None]),axis=0) # mean of lat grid lines
        lat_count = np.sum(iii,axis=0) # "length" of grid lines
        lat_res=np.mean(np.diff(lat_levels)) # Distance between parallels
        lat_pos = lat_mean//lat_res*lat_res + lat_res/2 # Move ticks from mean to between parallels
        
        # Add the longitudinal ticks
        [self.text(x,y,str(int(x)), horizontalalignment='center',verticalalignment='center') for x,y in zip(lon_levels[lat_count>count_min], lat_pos[lat_count>count_min])]
        
    
    def add_km_grid(self,resolution,**kwargs):
        '''
        Adds grid lines with equal physical distance on the grid

        Parameters
        ----------
        resolution : float
            Distance between the grid lines in km.
        **kwargs : 2D line properties.
            Passed to matplotlib.pyplot.plot

        Returns
        -------
        None.

        '''        
        
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
        '''
        Adds text to the cubed sphere projection

        Parameters
        ----------
        lon : float
            The geographic longitude to place the text.
        lat : float
            The geographic latitude to place the text.
        text : str
            The text.
        ignore_limits : bool, optional
            If True, text outside the plot limits are ignored. The default is False.
        **kwargs : text properties
            Passed to matplotlib.pyplot.text

        Returns
        -------
        Text
            The created Text instance.

        '''

        xi,eta = self.grid.projection.geo2cube(lon,lat)

        if self.grid.ingrid(lon,lat):
            return self.ax.text(xi, eta, text, **kwargs)
        else:
            print('text outside plot limit - set "ignore_limits = True" to override')
    
        
    def plot(self,lon,lat,**kwargs):
        '''
        Plots data points on the cubed sphere projection

        Parameters
        ----------
        lon : array-like or scalar
            The longitudinal coordinates of the data points.
        lat : array-like or scalar
            The latitudinal coordinates of the data points.
        **kwargs : 2D line properties
            Passed to matplotlib.pyplot.plot.

        Returns
        -------
        list of Line2D
            A list of lines representing the plotted data.

        '''
        x,y = self.grid.projection.geo2cube(lon,lat)
        return self.ax.plot(x,y,**kwargs)
    
    def scatter(self,lon,lat,**kwargs):
        '''
        Scatter plot of data points on the cubed sphere projection

        Parameters
        ----------
        lon : array-like or scalar
            The longitudinal coordinates of the data points.
        lat : array-like or scalar
            The latitudinal coordinates of the data points.
        **kwargs : 2D line properties
            Passed to matplotlib.pyplot.plot.

        Returns
        -------
        list of Line2D
            A list of lines representing the plotted data.

        '''
        x,y = self.grid.projection.geo2cube(lon,lat)
        return self.ax.scatter(x,y,**kwargs)

    def contour(self,*args,**kwargs):
        '''
        Plot contour lines on the cubes sphere projection.
        Call signature: contour([X, Y,] Z, **kwargs)

        Parameters
        ----------
        *args : Arrays
            X, Y : array-like, optional
                The lon,lat coordinates of the values in Z.
                X and Y must both be 2D with the same shape as Z
            Z : (M, N) array-like
                The height values over which the contour is drawn.
                Must be of self.grid.size if X,Y are not provided

        **kwargs : dict
            Passed to matplotlib.pyplot.contour.

        Returns
        -------
        QuadContourSet
            A set of contour lines or filled regions.

        '''
        
        if len(args)==1: # Only C provided
            return self.ax.contour(self.grid.xi,self.grid.eta,args[0],**kwargs)
        elif len(args)==3:
            X,Y = self.grid.projection.geo2cube(args[0],args[1])
            return self.ax.contour(X,Y,args[2],**kwargs)
        else:
            raise TypeError('Only accepts 1 or 3 arguments')         

    def contourf(self,*args,**kwargs):
        '''
        Plot filled contours on the cubes sphere projection.
        Call signature: contourf([X, Y,] Z, **kwargs)

        Parameters
        ----------
        *args : Arrays
            X, Y : array-like, optional
                The lon,lat coordinates of the values in Z.
                X and Y must both be 2D with the same shape as Z
            Z : (M, N) array-like
                The height values over which the filled contour is drawn.
                Must be of self.grid.size if X,Y are not provided

        **kwargs : dict
            Passed to matplotlib.pyplot.contourf.

        Returns
        -------
        QuadContourSet
            A set of contour lines or filled regions.

        '''
        
        if len(args)==1: # Only C provided
            return self.ax.contourf(self.grid.xi,self.grid.eta,args[0],**kwargs)
        elif len(args)==3:
            X,Y = self.grid.projection.geo2cube(args[0],args[1])
            return self.ax.contourf(X,Y,args[2],**kwargs)
        else:
            raise TypeError('Only accepts 1 or 3 arguments')     
     
    def pcolormesh(self,*args,**kwargs):
        '''
        Create a pseudocolor plot with a non-regular rectangular grid.
        Call signature: contourf([X, Y,] Z, **kwargs)
        
        Parameters
        ----------
        *args : Arrays
            X, Y : array-like, optional
                The lon,lat coordinates of the corners of the values in Z.
                X and Y must both be 2D with shape (M+1,N+1)
            Z : (M, N) array-like
                The values to be plotted.
                Must be of self.grid.size if X,Y are not provided
        
        **kwargs : dict
            Passed to matplotlib.pyplot.pcolormesh.
        
        Returns
        -------
        matplotlib.collections.QuadMesh
            A QuadMesh object.

        '''
        
        
        if len(args)==1: # Only C provided
            return self.ax.pcolormesh(self.grid.xi_mesh,self.grid.eta_mesh,args[0],**kwargs)
        elif len(args)==3:
            X,Y = self.grid.projection.geo2cube(args[0],args[1])
            return self.ax.pcolormesh(X,Y,args[2],**kwargs)
        else:
            raise TypeError('Only accepts 1 or 3 arguments') 
        
        
    def add_coastlines(self, resolution='110m' ,**kwargs):
        '''
        Adds coastlines to the cubed sphere projection.

        Parameters
        ----------
        resolution : str, optional
            DESCRIPTION. The default is '110m'.
        **kwargs : 2D line properties
            Passed to matplotlib.pyplot.plot.

        Returns
        -------
        None.

        '''
         
        if 'color' not in kwargs.keys():
             kwargs['color'] = 'black'
 
    
        for cl in self.grid.projection.get_projected_coastlines(resolution = resolution):
            xi, eta = cl
            self.ax.plot(xi, eta, **kwargs)
            

def anglemean(X,axis=None):
    '''
    Function to calculate the circular mean. NaNs are ignored.

    Parameters
    ----------
    X : array_like
        Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.

    Returns
    -------
    ndarray
        A new array containing the mean values

    '''
    return np.rad2deg(np.arctan2(np.nanmean(np.sin(np.deg2rad(X)),axis=axis),np.nanmean(np.cos(np.deg2rad(X)),axis=axis)))
    