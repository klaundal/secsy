import numpy as np
import secsy
import xarray as xr
from viresclient import SwarmRequest # for downloading Swarm data
from datetime import datetime
import matplotlib.pyplot as plt
import apexpy
import ppigrf # for geocentric->geodetic conversion

MINLON, MAXLON, MINLAT, MAXLAT = 197, 300, 41, 75 # magnetometers are all within this area
RE = 6371.2e3
lat0, lon0 = 58, 249             # location of grid
RI = RE + 110e3                  # ionosphere radius [m]
size_n, size_e = 3500e3, 6000e3  # approximate (!) meridional and zonal size of grid
res_n, res_e = 100e3, 100e3      # grid cell resolution
projection = secsy.cubedsphere.CSprojection((lon0, lat0), 0)
grid = secsy.cubedsphere.CSgrid(projection, size_e, size_n, res_e, res_n, R = RI)

fn = '/Users/laundal/Dropbox/science/data/induction/north_america_magnetometer_data.netcdf'

data = xr.open_dataset(fn)
ts = (data.time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
t0 = datetime.utcfromtimestamp(ts.values[ 0])
t1 = datetime.utcfromtimestamp(ts.values[-1])


# set up inversion
# ----------------
#    Differentiation matrix:
apx = apexpy.Apex(refh = 110, date = t0)
f1, f2 = apx.basevectors_qd(grid.lat.flatten(), grid.lon.flatten(), height = (RI - RE) * 1e-3, coords = 'geo')
De, Dn = grid.get_Le_Ln()
f1 = f1 / np.linalg.norm(f1, axis = 0)
Dem = De*f1[0].reshape((-1, 1)) + Dn*f1[1].reshape((-1, 1)) # calculates gradient in magnetic east direction
LTL = Dem.T.dot(Dem)

#    Design matrix
GBe, GBn, GBu = secsy.get_SECS_B_G_matrices(data.lat.values, data.lon.values, RE, grid.lat, grid.lon, current_type = 'divergence_free', RI = RI)
G = np.vstack((GBe, GBn, GBu)) # (3 * N_max) x grid.size
GTG = G.T.dot(G)

#    Regularization 
gtg_mag = np.median(GTG.diagonal())
ltl_mag = np.median(LTL.diagonal())
l1, l2 = 0.1, 0.1 # regularizatoin parameters
Ginv = np.linalg.pinv(GTG + l1*gtg_mag * np.eye(GTG.shape[0]) + l2 * gtg_mag / ltl_mag * LTL, rcond = 0)

#    Data matrix:
d = np.vstack((data.Be.values, data.Bn.values, -data.Bz.values)) # (3 * N_max) x N_timesteps
GTd = G.T.dot(d)


# Solve and calculate model predictions:
# --------------------------------------
#     Solve
m = Ginv.dot(GTd) # grid.size x N_timesteps

#     Calculate model predictions:
dm = G.dot(m)

#     Get Swarm data within grid
sampling_step = 'PT15S'
Swarm_data = {}
print('warning: The longitude filter used here will easily break if used in a different region')
for Swarm_satellite in ['A', 'B']:
    request = SwarmRequest()
    collection = 'SW_OPER_MAG' + Swarm_satellite + '_LR_1B'
    request.set_collection(collection)
    request.set_products(measurements=['F', 'B_NEC'], models=['CHAOS-Core'], residuals=False, sampling_step=sampling_step)
    request.set_range_filter('Latitude' , MINLAT, MAXLAT)
    request.set_range_filter('Longitude', MINLON - 360, MAXLON - 360)

    df = request.get_between(t0, t1).as_dataframe()
    _ = np.ones(len(df))

    # calculate a "geodetic radius": RE + geodetic height
    gdlat, h, __, __ = ppigrf.ppigrf.geoc2geod(90 - df['Latitude'].values, df['Radius'].values * 1e-3, _, _)
    df['R_gd'] = RE + h*1e3

    # Calculate magnetic field at Swarm altitude (very wasteful algorithm...)
    _Ge, _Gn, _Gu = secsy.get_SECS_B_G_matrices(gdlat, df['Longitude'].values, df['R_gd'].values, grid.lat, grid.lon, current_type = 'divergence_free', RI = RI)
    _Be, _Bn, _Bu = _Ge.dot(m), _Gn.dot(m), _Gu.dot(m)

    # The columns in these arrays refer to the times of the ground mags (every 1 min). Interpolate to Swarm time:
    Bexr = xr.DataArray(_Be, dims=["space", "time"], coords={"time": data.time.values})
    _Be = Bexr.interp(time = df.index).values.diagonal() # we only need the diagonal values (hence wasteful)
    Bnxr = xr.DataArray(_Bn, dims=["space", "time"], coords={"time": data.time.values})
    _Bn = Bnxr.interp(time = df.index).values.diagonal()
    Buxr = xr.DataArray(_Bu, dims=["space", "time"], coords={"time": data.time.values})
    _Bu = Buxr.interp(time = df.index).values.diagonal()

    # add CHAOS to get model F (** MAYBE BETTER CONVERT TO GEODETIC? **):
    BNEC_chaos = np.vstack(df["B_NEC_CHAOS-Core"].values).T
    F_chaos = np.linalg.norm(BNEC_chaos, axis = 0)
    _Be, _Bn, _Bu = _Be + BNEC_chaos[1], _Bn + BNEC_chaos[0], _Bu - BNEC_chaos[2]
    F = np.sqrt(_Be**2 + _Bn**2 + _Bu**2)

    # ground mag predicted delta F at Swarm orbit:
    df['dF_pred'] = F - F_chaos

    # delta F measured by Swarm:
    df['dF'] = df['F'] - F_chaos

    Swarm_data[Swarm_satellite] = df


# plot map and scatter plots
fig, axes = plt.subplots(ncols = 3, figsize = (20, 8))
cax = secsy.CSplot(axes[0], grid)
cax.scatter(data.lon.values, data.lat.values, c = 'C2')
cax.add_coastlines(resolution = '50m')

cax.scatter(Swarm_data['A'].Longitude, Swarm_data['A'].Latitude, marker = '.', s = 1, c = 'C0')
cax.scatter(Swarm_data['B'].Longitude, Swarm_data['B'].Latitude, marker = '.', s = 1, c = 'C1')

axes[1].scatter(Swarm_data['A']['dF'].values, Swarm_data['A']['dF_pred'].values, c = 'C0')
axes[2].scatter(Swarm_data['B']['dF'].values, Swarm_data['B']['dF_pred'].values, c = 'C1')
axes[1].set_title('Swarm A')
axes[2].set_title('Swarm B')

for i, ax in enumerate(axes[1:]):
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.plot([-300, 300], [-300, 300], color = 'C' + str(i))
    ax.set_aspect('equal')

    ax.set_xlabel('measured dF [nT]')
    ax.set_ylabel('model dF [nT]')

plt.tight_layout()

plt.show()



