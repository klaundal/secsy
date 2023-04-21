import lompe
import numpy as np
import matplotlib.pyplot as plt
from secsy.src.secsy import cubedsphere as cs

###################
# Script that testes the scalar_filed_interpoilation function against an analytic
# function 

#Make up a Lompe grid
position = (-30, 68)  # lon, lat for grid center
orientation = 0       # angle of grid x axis - anti-clockwise from east direction
L, W = 7000e3, 3800e3 # extents [m] of grid
dL, dW = 100e3, 100e3 # spatial resolution [m] of grid 
grid = cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, 
                       dL, dW, R = 6481.2e3)
shape = grid.shape

# Define analytic funtion to be able to compare performance to true value
def f(lon,lat):
    c1 = 70
    c2 = 40
    return np.sin(c1 * np.radians(lat)) + np.cos(c2 * np.radians(lon))
jpar = f(grid.lon, grid.lat)

# plt.pcolormesh(grid.lon, grid.lat, jpar.reshape(shape))

#input evaluation locations
eval_lat = np.linspace(51.9,57,100)
eval_lon = np.linspace(-35,-25, 50)
lon, lat = np.meshgrid(eval_lon, eval_lat)
xi_eval, eta_eval = grid.projection.geo2cube(lon, lat)
# inside = grid.ingrid(lon, lat)
# lon = lon[inside]
# lat = lat[inside]

#interpolated values
jpar_interp = grid.scalar_field_interpolation(lon, lat, jpar)

#### Test the performance
fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize =(14, 7))
ax[0].pcolormesh(grid.xi, grid.eta, jpar.reshape(grid.shape), vmin=0, vmax=2)
# ax[0].contour(xi_eval, eta_eval, jpar_interp.reshape(xi_eval.shape), levels=np.linspace(0,10,11), linewidths=3)
ax[0].set_title('Analytic, evaluated on CS grid')
ax[0].set_xlim(grid.xi_min, grid.xi_max)
ax[0].set_ylim(grid.eta_min, grid.eta_max)
ax[1].pcolormesh(xi_eval, eta_eval, jpar_interp.reshape(xi_eval.shape), vmin=0, vmax=2)
ax[1].scatter(grid.xi, grid.eta, c = jpar.reshape(grid.shape), vmin = 0, vmax = 2, s = 30, edgecolors = 'black', linewidths = .8)
ax[1].set_title('Interpolated, on fine grid')
ax[1].set_xlim(-0.05,0.05)
ax[1].set_ylim(-0.265,-0.235)

ax[2].pcolormesh(xi_eval, eta_eval, f(lon,lat).reshape(xi_eval.shape), vmin=0, vmax=2)
ax[2].set_title('Analytic, on fine grid')
ax[2].set_xlim(-0.05,0.05)
ax[2].set_ylim(-0.265,-0.235)
ax[2].scatter(grid.xi, grid.eta, c = jpar.reshape(grid.shape), vmin = 0, vmax = 2, s = 30, edgecolors = 'black', linewidths = .8)
