import numpy as np
import apexpy
import secsy
import ppigrf
import datetime as dt
import dask # used to parallelize the calculations
import dask.array as da
from pynamit import CSProjection
from pynamit import SHBasis, BasisEvaluator, Grid, Mainfield, FieldEvaluator
from scipy.io.matlab import savemat, loadmat
from pynamit.math.tensor_operations import tensor_pinv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

dask.config.set(num_workers = 8)

RE = 6371.2e3  # Earth radius in km
mu0 = 4 * np.pi * 1e-7

LOWLATFILTER = 20 # skip SECS functions equatorward of this
low_lat_boundary = 0 #  # below this boundary, hemispheres are coupled
height = 110e3  # height of the currents in km
upper_limit = 6.6 * RE  # upper limit of the integration
hs = np.logspace(np.log10(height), np.log10(upper_limit - RE), 51)
drs = np.diff(hs)
epoch = 2016.
date = dt.datetime(int(epoch), 3, 12, 3, 0)
RI = RE + height
N, M = 20, 20 # spherical harmonic truncation levels
apx = apexpy.Apex(epoch, height * 1e-3) # apex object used for field line mapping
mainfield = Mainfield(kind = 'igrf', epoch = int(epoch))

if low_lat_boundary > 0:
    print('low lat boundary is set > 0. For now this does not work with the SH part, only SECS') 


# set up SECS grid and observation grids
csp = CSProjection(20)
delta = np.diff(csp.arr_eta)[0]
pole_lat, pole_lon = 90 - csp.arr_theta, csp.arr_phi
__r, th, obs_lon = csp.cube2spherical(csp.arr_xi + delta/2, csp.arr_eta + delta/2, block = csp.arr_block, deg = True)
obs_lat = 90 - th



# SET UP THE SPHERICAL HARMONICS PART

# set up random spherical harmonics for mutual validation of SECS and SH approaches
shbasis = SHBasis(N, M)
np.random.seed(42) 
T_coeffs  = np.random.random(shbasis.n.size) - .5
T_coeffs  =  T_coeffs / (shbasis.n * (shbasis.n + 1))**2 # some kind of scaling to make the higher order coefficints smaller
jr_coeffs = -T_coeffs * (shbasis.n ** 2 + shbasis.n) / (RI * mu0) # converted to jr coefficients


# get spherical harmonic evaluator
shgrid   = Grid(lat = obs_lat , lon = obs_lon)
polegrid = Grid(lat = pole_lat, lon = pole_lon)
obspoint_evaluator = BasisEvaluator(shbasis, shgrid  ) # SH evaluator for observation points
polpoint_evaluator = BasisEvaluator(shbasis, polegrid) # SH evaluator for pole locations


# the scale factor used to get Br on ground:
Ve_to_Brground = -shbasis.n * (RE / RI) ** (shbasis.n - 1)

# calculate matrix that produces poloidal part of FAC magnetic field from toroidal coefficients

Pe_to_JS = polpoint_evaluator.G_rxgrad * (2 * shbasis.n + 1) / (shbasis.n + 1) / mu0 
JS_to_Pe = tensor_pinv(Pe_to_JS, contracted_dims = 2, rtol = 0)

T_to_Pe_r = np.zeros((shbasis.index_length, shbasis.index_length)) # matrix that transforms from T coefficients to coeffs for external potential

rks = hs[:-1] + drs/2 + RE
for i, rk in enumerate(rks):
    print(f'Calculating matrix for poloidal field of inclined FACs. Progress: {i+1}/{rks.size}', end = '\r' if i < (rks.size - 1) else '\n', flush = True)
    # Map coordinates from rk to RI:
    theta_mapped, phi_mapped = mainfield.map_coords(RI, rk, polegrid.theta, polegrid.phi)
    mapped_grid = Grid(theta = theta_mapped, phi = phi_mapped)

    # Matrix that gives jr at mapped grid from toroidal coefficients, shifts to rk[i], and extracts horizontal current components
    rk_b_evaluator = FieldEvaluator(mainfield, polegrid, rk)
    mapped_b_evaluator = FieldEvaluator(mainfield, mapped_grid, RI)
    mapped_basis_evaluator = BasisEvaluator(shbasis, mapped_grid)
    T_to_jr     = mapped_basis_evaluator.scaled_G(-shbasis.n * (shbasis.n + 1) / (RI * mu0) )
    jr_to_JS_rk = np.array([rk_b_evaluator.Btheta / mapped_b_evaluator.Br,
                            rk_b_evaluator.Bphi   / mapped_b_evaluator.Br])

    T_to_JS_rk = np.einsum('ij,jk->ijk', jr_to_JS_rk, T_to_jr, optimize = True)

    # Matrix that calculates the contribution to the poloidal coefficients from the horizontal current components at rk[i]
    Pe_rk_to_Pe_r = shbasis.radial_shift(rk, RI).reshape((-1, 1, 1))
    JS_rk_to_Pe_r = JS_to_Pe * Pe_rk_to_Pe_r

    # Integration step
    T_to_Pe_r += drs[i] * np.tensordot(JS_rk_to_Pe_r, T_to_JS_rk, 2)


pol_coeffs_from_fac = T_to_Pe_r.dot(T_coeffs)


Br_radial_sh = obspoint_evaluator.basis_to_grid(pol_coeffs_from_fac * Ve_to_Brground)




# SECS PART

# start by calculating the SECS current amplitudes using the SH representation
jr_pole = polpoint_evaluator.basis_to_grid(jr_coeffs) # radial current density
Ir_pole = jr_pole * csp.unit_area*RI**2 # the radial current



shape = (obs_lat.size, pole_lat.size)

r_obs = RE * np.vstack((np.cos(np.deg2rad(obs_lat)) * np.cos(np.deg2rad(obs_lon)),
                        np.cos(np.deg2rad(obs_lat)) * np.sin(np.deg2rad(obs_lon)),
                        np.sin(np.deg2rad(obs_lat))))


pole_mlat, pole_mlon = apx.geo2apex(pole_lat, pole_lon, height * 1e-3)

ll = np.abs(pole_mlat) < low_lat_boundary  # index of low latitude points that are to be connected

# include poles from the opposite hemisphere at low latitudes:
pole_lat_cp, pole_lon_cp, _ = apx.apex2geo(-pole_mlat[ll], pole_mlon[ll], height * 1e-3)
pole_lat = np.hstack((pole_lat, pole_lat_cp))
pole_lon = np.hstack((pole_lon, pole_lon_cp))
pole_mlat = np.hstack((pole_mlat, -pole_mlat[ll]))
pole_mlon = np.hstack((pole_mlon,  pole_mlon[ll]))
extended_shape = (obs_lat.size, pole_lat.size)

apex_height = apx.get_apex(pole_mlat) * 1e3

print('todo: move r_current up to midway point in integral segment  -- but this must be accounted for when adding the radial line')
@dask.delayed
def compute_G_components(h, dr):
    iii = (apex_height >= h) & (np.abs(pole_mlat) > LOWLATFILTER)
    if not np.any(iii):
        return np.zeros(extended_shape), np.zeros(extended_shape), np.zeros(extended_shape)
    
    glat, glon = np.empty_like(pole_mlat), np.empty_like(pole_mlat)
    glat[iii], glon[iii], _ = apx.apex2geo(pole_mlat[iii], pole_mlon[iii], h * 1e-3)
    r_current = (RE + h) * np.vstack((np.cos(np.deg2rad(glat[iii])) * np.cos(np.deg2rad(glon[iii])),
                                      np.cos(np.deg2rad(glat[iii])) * np.sin(np.deg2rad(glon[iii])),
                                      np.sin(np.deg2rad(glat[iii]))))
    r_diff = r_current.reshape((3, -1, 1)) - r_obs.reshape((3, 1, -1))

    B = np.vstack(ppigrf.igrf(glon[iii], glat[iii], h * 1e-3, date))
    b = B / np.linalg.norm(B, axis=0)
    b[:, b[2] < 0] *= -1 # make all dl's point upward
    dl = dr / b[2]  # length of this segment

    # figure out if the next step in the integral is above the field line apex - if it is, we need to cut the line segment and add a radial line current at the end to close the circuit in infinity
    jjj = apex_height[iii] <= (h + dr) # True if next point is above apex, False otherwise
    #print(dl.shape, jjj.shape, apex_height.shape, iii.shape, dr, h)
    assert np.all(apex_height[iii][jjj] - h > 0)
    dl[jjj] = dl[jjj] / dr * (apex_height[iii][jjj] - h) # shorten dl where appropriate

    bxr_diff = np.cross(b.reshape((3, -1, 1)), r_diff, axisa = 0, axisb = 0, axisc = 0)

    #print(bxr_diff.shape, r_diff.shape, dl.shape)
    Gx_, Gy_, Gz_ = mu0 / (4 * np.pi) * bxr_diff / np.linalg.norm(r_diff, axis=0) ** 3 * dl.reshape((1, -1, 1))
    Gx, Gy, Gz = np.zeros(extended_shape), np.zeros(extended_shape), np.zeros(extended_shape)
    #print(Gx_.shape, dl.shape, bxr_diff.shape, dl.reshape((1, -1, 1)).shape)
    Gx[:, iii] = Gx_.T
    Gy[:, iii] = Gy_.T
    Gz[:, iii] = Gz_.T

    
    if sum(jjj) > 0: # add the magnetic field of an infinite radial line current at the end of the integral path if the path goes beyond apex
        #print(r_diff.shape, dl.shape, jjj.shape)
        #print(r_diff[:, jjj, :].shape, dl[jjj].shape)
        #print(r_current[:, jjj].shape, r_diff[:, jjj, :].shape, dl[jjj].reshape(1, -1).shape, ' <--')
        r_line_base = r_current[:, jjj] + b[:, jjj] * dl[jjj].reshape((1, -1)) # ecef coordinates of the line base

        # calculate the spherical coordinates of the base locations, and use secsy to get the matrices:
        #print(r_line_base.shape)
        _r, _th, _ph = secsy.spherical.car_to_sph(r_line_base)
        _Ge_radial, _Gn_radial, _Gu_radial = secsy.utils.get_wedge_G_matrix(obs_lat, obs_lon, np.full_like(obs_lat, RE), 90 - _th, _ph, _r, np.zeros_like(_r), np.zeros_like(_r), np.ones_like(_r), calculate_radial_leg = False)

        # calculate ECEF matrices (eq 4 in Laundal & Richmond)
        _th, _ph = np.deg2rad(_th), np.deg2rad(_ph)
        _Gx_radial = -_Ge_radial * np.sin(_ph) - _Gn_radial * np.cos(_th) * np.cos(_ph) + _Gu_radial * np.sin(_th) * np.cos(_ph)
        _Gy_radial =  _Ge_radial * np.cos(_ph) - _Gn_radial * np.cos(_th) * np.sin(_ph) + _Gu_radial * np.sin(_th) * np.sin(_ph)
        _Gz_radial =                             _Gn_radial * np.sin(_th)               + _Gu_radial * np.cos(_th)

        # add the result to the main matrices:
        #print(Gx.shape, _Gx_radial.shape, iii.shape, jjj.shape, np.sum(iii), np.sum(jjj))
        Gx[:, iii][:, jjj] += _Gx_radial
        Gy[:, iii][:, jjj] += _Gy_radial
        Gz[:, iii][:, jjj] += _Gz_radial
    
 
    return Gx, Gy, Gz
    


# Process results in chunks to reduce memory overhead
chunk_size = 20  # Adjust based on available memory
Gx, Gy, Gz = da.zeros(extended_shape), da.zeros(extended_shape), da.zeros(extended_shape)

print(dt.datetime.now())
for i in range(0, len(hs) - 1, chunk_size):
    chunk_results = [compute_G_components(h, dr) for h, dr in zip(hs[i:i+chunk_size], drs[i:i+chunk_size])]
    chunk_computed = dask.compute(*chunk_results)
    
    Gx += sum(x for x, _, _ in chunk_computed)
    Gy += sum(y for _, y, _ in chunk_computed)
    Gz += sum(z for _, _, z in chunk_computed)

    print(i, (len(hs)-1))


# The matrices are to be multiplied by the total current, not the radial component, so we need to scale them by 1/b_r at the footpoint
B  = np.vstack(ppigrf.igrf(pole_lon, pole_lat, height * 1e-3, date))
b  = B / np.linalg.norm(B, axis=0)
br = np.abs(b[2].reshape((1, -1)))
Gx, Gy, Gz = Gx/br, Gy/br, Gz/br

print("Dask parallel computation complete.", dt.datetime.now())


# convert from Cartesian to enu
ph, th = np.deg2rad(obs_lon).reshape((-1, 1)), np.deg2rad(90 - obs_lat).reshape((-1, 1))
Ge_igrf_fieldline = -np.sin(ph)              * Gx + np.cos(ph)              * Gy
Gn_igrf_fieldline = -np.cos(th) * np.cos(ph) * Gx - np.cos(th) * np.sin(ph) * Gy + np.sin(th) * Gz
Gu_igrf_fieldline =  np.sin(th) * np.cos(ph) * Gx + np.sin(th) * np.sin(ph) * Gy + np.cos(th) * Gz


# Get magnetic field of the radial current:
Ge_radial, Gn_radial, Gu_radial = secsy.utils.get_wedge_G_matrix(   obs_lat, obs_lon, np.full_like(obs_lat, RE), pole_lat, pole_lon, np.full_like(pole_lon, RI), np.zeros_like(pole_lon), np.zeros_like(pole_lon), np.ones_like(pole_lon), calculate_radial_leg = False)

# Get the standard SECS matrix
#Ge_secs, Gn_secs, Gu_secs       = secsy.utils.get_SECS_B_G_matrices(obs_lat, obs_lon, np.full_like(obs_lat, RE), pole_lat, pole_lon, current_type = 'curl_free', RI = RI)

# Combine all the matrices
Ge_ = Ge_igrf_fieldline - Ge_radial #+ Ge_secs
Gn_ = Gn_igrf_fieldline - Gn_radial #+ Gn_secs
Gu_ = Gu_igrf_fieldline - Gu_radial #+ Gu_secs

if any(ll):
    # The effects of the conjugate points should be added onto the originals:
    Ge = Ge_[:, :shape[1]]
    Gn = Gn_[:, :shape[1]]
    Gu = Gu_[:, :shape[1]]

    Ge[:, ll] = (Ge[:, ll] - Ge_[:, shape[1]:]).compute()
    Gn[:, ll] = (Gn[:, ll] - Gn_[:, shape[1]:]).compute()
    Gu[:, ll] = (Gu[:, ll] - Gu_[:, shape[1]:]).compute()
else:
    Ge = Ge_.compute()
    Gn = Gn_.compute()
    Gu = Gu_.compute()





# Add effect of shielding currents
print('TODO: Shielding current can be added by calculating Br in a CS grid and use that to work out what the DF SECS in each cell must be to compensate Br. Then use the magnetic field of those DF SECS functions to calculate the magnetic field at the desired evaluation points')




fig, axes = plt.subplots(figsize=(12, 12), nrows = 2, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axes.flatten():
    ax.set_global()
        
    # Add features
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
axes[0, 0].scatter(obs_lon, obs_lat, c = Gu.dot(Ir_pole), vmin = -0.05, vmax = 0.05, cmap = plt.cm.bwr)
axes[1, 1].scatter(obs_lon, obs_lat, c = Ge.dot(Ir_pole), vmin = -0.05, vmax = 0.05, cmap = plt.cm.bwr)
axes[0, 1].scatter(obs_lon, obs_lat, c = Br_radial_sh   , vmin = -0.02, vmax = 0.02, cmap = plt.cm.bwr)
axes[1, 0].scatter(obs_lon, obs_lat, c = Ir_pole, vmax = 7244126315, vmin = -7244126315, cmap = plt.cm.bwr)
axes[0, 0].set_title('Br SECS')
axes[1, 1].set_title('Be SECS')
axes[0, 1].set_title('Br SH')
axes[1, 0].set_title('Ir')

    
plt.show()




