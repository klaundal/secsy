""" test vector projections in cubed sphere coordinates """
import numpy as np
from secsy import get_SECS_J_G_matrices
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from importlib import reload
from pysymmetry.utils import cubedsphere
reload(cubedsphere)


def Geocentric_to_PlateCarree_vector_components(east, north, latitude):
    """ convert east north vector components to Plate Carree projection 

        parameters
        ----------
        east: array-like
            eastward components
        north: array-like
            westward components
        latitude: array-like
            latitude of each vector

        returns
        -------
        east, north in Plate Carree projection
        Preserveres shape and norm

    """
    magnitude = np.sqrt(east**2 + north**2)

    east_pc = east / np.cos(latitude * np.pi / 180)

    magnitude_pc = np.sqrt(east_pc**2 + north**2)

    east_pc  = east_pc * magnitude / magnitude_pc
    north_pc = north * magnitude / magnitude_pc

    return east_pc, north_pc



lat0, lon0 = 65., 15. # center of projection
e, n = 3, 3. # orientation of projection in east/north

projection = cubedsphere.CSprojection((lon0, lat0), (e, n))
grid = cubedsphere.CSgrid(projection, 3000, 2000, 100., 100.)

vlon, vlat = projection.cube2geo(grid.xi.flatten() + grid.dxi/2, grid.eta.flatten() + grid.deta/2)

Ge, Gn = get_SECS_J_G_matrices(vlat, vlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free')

# make synthetic model vector
m = np.zeros_like(grid.lon)
m[m.shape[0]//2, m.shape[1]//2] = 10
m = m.flatten()

je, jn = Ge.dot(m), Gn.dot(m)

vxi, veta = projection.vector_cube_projection(je, jn, vlon, vlat)

""" SET UP PLOT """
fig = plt.figure()
# CUBED SPHERE PROJECTION
axc = fig.add_subplot(121)
#for c in projection.get_projected_coastlines():
#    axc.plot(c[0], c[1], 'k-')
axc.set_xlim(grid.xi.min(), grid.xi.max())
axc.set_ylim(grid.eta.min(), grid.eta.max())
xi, eta = projection.geo2cube(lon0, lat0)
axc.scatter(xi, eta)
xi, eta = projection.geo2cube(vlon, vlat)

axc.quiver(xi, eta, vxi, veta)

# CARTOPY PROJECTION
proj = ccrs.LambertAzimuthalEqualArea(central_latitude = lat0, central_longitude = lon0)
axm = fig.add_subplot(122, projection = proj)


land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='face',
                                    facecolor='lightgreen')


axm.add_feature(land, zorder=0)
axm.gridlines()
axm.set_extent([lon0 - 7, lon0 + 7, lat0 - 7, lat0 + 7])
#axm.scatter(lon0, lat0, transform = ccrs.Geodetic(), c = 'red', marker = 'o', zorder = 5)

je_pc, jn_pc = Geocentric_to_PlateCarree_vector_components(je, jn, vlat)
axm.quiver(vlon, vlat, je_pc, jn_pc, transform = ccrs.PlateCarree())#, regrid_shape = (40, 30))


plt.show()
