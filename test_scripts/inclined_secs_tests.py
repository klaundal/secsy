""" Demonstrate / test calcualations of magnetic field from CF SECS systems with inclined field lines

    This script uses pyamps (pip install pyamps) and the dipole module which can be found here: https://github.com/klaundal/dipole
"""
import numpy as np
import matplotlib.pyplot as plt
from secsy import spherical, CSprojection, CSgrid, get_CF_SECS_B_G_matrices_for_inclined_field
import pyamps
import dipole 

d2r = np.pi / 180
RE = 6371.2 * 1e3

# load an AMPS current map and multiply current density by grid area to get SECS amplitude
a = pyamps.AMPS(400, 0, -4, 23, 100)
grid = CSgrid(CSprojection((0, 90), 0), 25000e3, 25000e3, 200e3, 200e3, R = RE + 110e3)
lat, mlt = grid.lat, grid.lon / 15
jr = a.get_upward_current(lat, mlt) * 1e-6 # A/m^2
Ir = jr * grid.A # SECS amplitudes

# get dipole field line orientations
d = dipole.Dipole(2020)
Bn, Br = d.B(lat, grid.R)

elat, elon = grid.lat_mesh, grid.lon_mesh # evaluation points
er = np.full_like(elat, RE) # evaluation radius
Ge, Gn, Gr = get_CF_SECS_B_G_matrices_for_inclined_field(elat.flatten(), elon.flatten(), er.flatten(), lat.flatten(), mlt.flatten()*15, Bn.flatten() * 0, Bn.flatten(), Br.flatten(), RI = grid.R)
Be, Bn, Br = Ge.dot(Ir.flatten()), Gn.dot(Ir.flatten()), Gr.dot(Ir.flatten())


fig = plt.figure(figsize = (14, 8))
ax1 = plt.subplot2grid((1, 35), (0, 0 ), colspan = 15)
ax2 = plt.subplot2grid((1, 35), (0, 16), colspan = 15)
cax = plt.subplot2grid((1, 35), (0, 34))

faclevels = np.linspace(-1.15, 1.15, 24)
ax1.contourf(grid.xi, grid.eta, jr * 1e6, cmap = plt.cm.bwr, levels = faclevels)
ax1.contour(grid.xi, grid.eta, grid.lat, levels = [50, 60, 70, 80], colors = 'black', linestyles = '--', linewidths = .5)
ax1.set_aspect('equal')
ax1.set_axis_off()
ax1.set_title('FAC')

Blevels = np.linspace(-11.5, 11.5, 24)
ax2.contourf(grid.xi_mesh, grid.eta_mesh, Br.reshape((grid.xi_mesh.shape)) * 1e9, cmap = plt.cm.bwr, levels = Blevels)
ax2.contour(grid.xi, grid.eta, grid.lat, levels = [50, 60, 70, 80], colors = 'black', linestyles = '--', linewidths = .5)

xi, eta, Bxi, Beta = grid.projection.vector_cube_projection(Be, Bn, grid.lon_mesh.flatten(), grid.lat_mesh.flatten())
q = ax2.quiver(xi, eta, Bxi * 1e9, Beta * 1e9, units = 'width')
ax2.quiverkey(q, 0.8, 1.05, 10, r'10 nT', labelpos='E', coordinates='axes')
ax2.set_title('Ground magnetic field')

ax2.set_aspect('equal')
ax2.set_axis_off()


cax.contourf(np.vstack((np.zeros_like(faclevels), np.ones_like(faclevels))).T, np.vstack((faclevels, faclevels)).T, np.vstack((faclevels, faclevels)).T, cmap  =plt.cm.bwr, levels= faclevels)
cax.set_ylabel(r'upward FAC [$\mu$A/m$^2$]', size = 14)
cax.set_xticks([])
caxB = cax.twinx()
caxB.set_ylim(Blevels.min(), Blevels.max())
caxB.set_ylabel('radial magnetic field [nT]', size = 14)

plt.savefig('figures/magnetic_field_of_FACs_on_inclined_field_lines.png', dpi = 200)



#####################################################################################
# 2D plot of cross section of Earth including two currents along inclined field lines

RI = RE + 500e3
minlat = 40 # lowest latitude of the plot
clat = 60 # latitude of the two currents

# set up the grid
x = np.linspace(-RE * np.cos(minlat * d2r), RE * np.cos(minlat * d2r), 160)
z = np.linspace( RE/2, 1.5 * RE, 160)
xx, zz = np.meshgrid(x, z)

# calculate magnetic field orientations, and set up current system parameters
I_n0 = np.array([-np.cos(clat * d2r)   ] * 2)  # Dipole orientations
I_r  = np.array([2 * np.sin(clat * d2r)] * 2)  # 
clat = np.array([clat] * 2)
clon = np.array([180, 0]) # longitudes of the currents
I    = np.array([-1, 1]) * 100e3 # current magnitude in Amperes

# calculate evaluation points in spherical coordinates
rr = np.sqrt(xx**2 + zz**2)
la = np.arccos(np.abs(xx) / rr) / d2r
lo = np.arctan2(0, xx) / d2r

counter = 0 # loop through several orientations of the SECS current:
for I_n in (I_n0.reshape((1, 2)) + I_n0.reshape((1, 2)) * np.linspace(-2, 2, 33).reshape((33, 1))):

    # get the matrices
    Ge, Gn, Gr = get_CF_SECS_B_G_matrices_for_inclined_field(la.flatten(), lo.flatten(), rr.flatten(), clat, clon, np.zeros_like(I_n), I_n, I_r, RI = RI)
    Be = Ge.dot(I)
    Bn = Gn.dot(I)
    Br = Gr.dot(I)

    Bx, By, Bz = spherical.enu_to_ecef(np.vstack((Be, Bn, Br)).T, lo.flatten(), la.flatten()).T
    Bx, By, Bz = Bx.reshape(xx.shape), By.reshape(xx.shape), Bz.reshape(xx.shape)

    fig = plt.figure(figsize = (14, 8))
    ax = plt.subplot2grid((1, 20), (0, 0), colspan = 18)
    cax = plt.subplot2grid((1, 20), (0, 19))
    levels = np.linspace(-20, 20, 32)*1e-9

    ax.contourf(xx, zz, By, cmap = plt.cm.bwr, levels = levels, extend = 'both')
    ax.contour(xx, zz, rr, levels = [RE, RI], colors = 'black')
    ax.contour(xx, zz, rr, levels = [RI], colors = 'black', linewidths = 5)


    # plot the current path 
    s = spherical.enu_to_ecef(np.vstack((np.zeros_like(I_n), I_n, I_r)).T, clon, clat).T # ecef vectors along current
    x1, y1 = RI * np.cos(clat * d2r) * np.cos(clon * d2r), RI * np.sin(clat * d2r)
    x2 = np.array([x.min(), x.max()])
    y2 = y1 + s[2] * (x2 - x1) / s[0]
    ax.plot(np.vstack((x1, x2)), np.vstack((y1, y2)), 'k-', linewidth = 5)

    # plot radial line for comparison
    s = spherical.enu_to_ecef(np.vstack((np.zeros_like(I_n), I_n*0, I_r)).T, clon, clat).T # ecef vectors along current
    x2 = np.array([x.min(), x.max()])
    y2 = y1 + s[2] * (x2 - x1) / s[0]
    ax.plot(np.vstack((x1, x2)), np.vstack((y1, y2)), 'k--', linewidth = 1)


    cax.contourf(np.vstack((np.zeros_like(levels), np.ones_like(levels))).T, np.vstack((levels, levels)).T * 1e9, np.vstack((levels, levels)).T * 1e9, cmap  =plt.cm.bwr, levels= levels * 1e9)
    cax.set_ylabel('Magnetic field out of plane [nT] (saturated)', size = 14)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")




    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.min(), z.max())

    plt.savefig('figures/' + str(counter).zfill(2) + '.png', dpi = 200)
    if np.isclose(I_n[0], I_n0[0]):
        plt.savefig('figures/dipole.png', dpi = 200)
    if np.isclose(I_n[0], 0):
        plt.savefig('figures/radial.png', dpi = 200)

    counter += 1



