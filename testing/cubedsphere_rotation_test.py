""" Script to test how precisely the L matrices in CSgrid calcualte partial derivatives of
    a function defined on the sphere

    I'm using spherical harmonics to do this test
"""
import numpy as np
from secsy import CSgrid, CSprojection
import matplotlib.pyplot as plt

### SET UP CUBED SPHERE GRID AND PROJECTION
position, orientation = (0, 90), (0, -1)
projection = CSprojection(position, orientation)
grid = CSgrid(projection, 5000, 5000, 12., 10.)

llon, llat = grid.local_lon[::15, ::15].flatten(), grid.local_lat[::15, ::15].flatten()
glon, glat = grid.lon      [::15, ::15].flatten(), grid.lat      [::15, ::15].flatten()
N = glon.size
east, north = np.zeros(N), np.ones(N)

xi, eta, Axi, Aeta = grid.projection.vector_cube_projection(east, north, glon, glat)

fig, axes = plt.subplots(ncols = 2)
axes = axes.flatten()
axes[0].quiver(xi, eta, Axi, Aeta, scale = 6.)

R = grid.projection.local2geo_enu_rotation(llon, llat)
l_e, l_n = np.einsum('nij, jn -> in', R, np.vstack((east, north)))

xi, eta, Axi, Aeta = grid.projection.vector_cube_projection(l_e, l_n, glon, glat)
axes[1].quiver(xi, eta, Axi, Aeta, scale = 6.)


print(l_e**2 + l_n**2)

for ax in axes:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    #for c in projection.get_projected_coastlines():
    #    ax.plot(c[0], c[1], 'k-', linewidth = .4)
    for lat in np.r_[-80:81:10]:
        x, y = projection.geo2cube(np.arange(361), np.ones(361) * lat)
        ax.plot(x, y, 'k:')
    for lon in np.r_[0:361:15]:
        x, y = projection.geo2cube(np.ones(180) * lon, np.linspace(-80, 80, 180))
        ax.plot(x, y, 'k:')
    #ax.scatter(grid.xi.flatten(), grid.eta.flatten(), s = 1, c = 'black')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

plt.ion()
plt.show()
plt.pause(0.001)
