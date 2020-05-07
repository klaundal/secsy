""" Code for working with cubed sphere projection in in a limited region. 
    A cubed sphere grid is a grid that is defined via the projection of a circumscribed 
    cube onto a sphere. The great advantage of this grid is that it avoids any pole 
    problem, and that there is not a large variation in spatial resolution across the 
    grid. The disadvantage is that it is non-orthogonal, which means that differential 
    operators change. The purpose of this script is to take care of that problem.

    This code only implements a grid on (part of) one side of the cube. The purpose
    is to use it for regional data analyses such as SECS, and potentially simple
    modelling. The code uses the equations for the north pole side of the cube,
    except for a reversal in signs of the cube coordinates xi, eta

    The grid and associated math is completely based on:
    C. Ronchi, R. Iacono, P.S. Paolucci, The “Cubed Sphere”: A New Method for the 
    Solution of Partial Differential Equations in Spherical Geometry, Journal of 
    Computational Physics, Volume 124, Issue 1, 1996, Pages 93-114, 
    https://doi.org/10.1006/jcph.1996.0047.

    KML, May 2020
"""

import numpy as np
d2r = np.pi / 180
from secsy import spherical
import cartopy.io.shapereader as shpreader

class CSprojection(object):
    def __init__(self, position, orientation):
        """ Set up cubed sphere projection

        The CSprojection is set up by 
        1) rotating to a local coordinate system in which 'position' 
        is at the pole, and 'orientation' defines the x axis (prime meridian)
        2) applying the Ronchi et al. conversions to xi, eta coords on the 
        local coordinates

        Parameters
        ----------
        position: array (lon, lat)
            coordinate at which the cube surface should be 
            tangential to the sphere - the center of the projection.
            Pair of values for longitude and latitude [deg]
        orientation: array (east, north)
            orientation of the cube surface - a 2D vector defined by
            its geocentric (east, north) components. This direction
            defines the direction of constant xi (i.e. the eta axis) 
        """

        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.orientation = self.orientation / np.linalg.norm(orientation) # normalize

        self.lon0, self.lat0 = position

        # the z axis of local coordinat system described in geocentric coords:
        self.z = np.array([np.cos(self.lat0 * d2r) * np.cos(self.lon0 * d2r), 
                           np.cos(self.lat0 * d2r) * np.sin(self.lon0 * d2r),
                           np.sin(self.lat0 * d2r)])

        # the x axis is the orientation described in ECEF coords:
        self.x = spherical.enu_to_ecef(np.array([self.orientation[0], self.orientation[1], 0]).reshape((1, 3)), np.array(self.lon0), np.array(self.lat0)).flatten()
        
        # the y axis completes the system:
        self.y = np.cross(self.z, self.x)
 
        # define rotation matrices for rotations between local and geocentric:
        self.R_geo2local = np.vstack((self.x, self.y, self.z)) # rotation matrix from GEO to rotated coords (ECEF)
        self.R_local2geo = self.R_geo2local.T  # inverse


    def geo2cube(self, lon, lat):
        """ convert from geocentric coordinates to cube coords (xi, eta) 
        
        Input parameters must have same shape. Output will have same shape.
        Points that are outside the cube surface will be nans   

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al.(*), after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]
        eta: array
            eta, as defined in Ronchi et al. (*), after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]

        Note
        ----
        (*) the signs of xi and eta are reversed compared to Ronchi et al., 
        so that eta is positive along self.orientation
        """

        lon, lat = np.array(lon), np.array(lat)
        shape = lon.shape
        lon, lat = lon.flatten(), lat.flatten()

        # first convert to local spherical coordinate system (ROT):
        lon, lat = self.geo2local(lon, lat)

        theta, phi = (90 - lat) * d2r, lon * d2r
        X =  np.tan(theta) * np.sin(phi)
        Y = -np.tan(theta) * np.cos(phi)

        xi, eta = np.arctan(X), np.arctan(Y)

        # mask elements outside cube surface by nans:
        ii = theta > np.pi/4
        xi [ii] = np.nan
        eta[ii] = np.nan
        return -xi.reshape(shape), -eta.reshape(shape)


    def cube2geo(self, xi, eta):
        """ Convert from cube coordinates (xi, eta) to geocentric (lon, lat)

        Input parameters must have same shape. Output will have same shape.
        Points that are outside the cube surface will be nans   

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al. (*), after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]
        eta: array
            eta, as defined in Ronchi et al. (*), after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]

        Note
        ----
        (*) the signs of xi and eta are reversed compared to Ronchi et al., 
        so that eta is positive along self.orientation

        """
        xi, eta = -np.array(xi), -np.array(eta)
        shape = xi.shape
        xi, eta = xi.flatten(), eta.flatten()

        X = np.tan(xi)
        Y = np.tan(eta)
        phi = -np.arctan(X / Y)
        theta = np.arctan(X / np.sin(phi))

        lon, lat = self.local2geo(phi / d2r, 90 - theta / d2r)

        return lon.reshape(shape), lat.reshape(shape)


    def geo2local(self, lon, lat, reverse = False):
        """ Convert from geocentric coordinates to local coordinates 

        lon and lat must have the same shape. Shapes are preserved in output.

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]
        reverse: bool, optional
            set to False (default) if you want to rate from geocentric to local, 
            set to True if you want the opposite rotation

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system
        """
        assert lat.shape == lon.shape
        shape = lat.shape

        # set up ECEF position vectors, and rotate using rotation matrices
        lat, lon = np.array(lat).flatten() * d2r, np.array(lon).flatten() * d2r
        r = np.vstack((np.cos(lat) * np.cos(lon), 
                       np.cos(lat) * np.sin(lon),
                       np.sin(lat)))
        if reverse:
            r_ = self.R_local2geo.dot(r)
        else:
            r_ = self.R_geo2local.dot(r)

        # calcualte spherical coords:
        newlat = np.arcsin (r_[2]) / d2r
        newlon = np.arctan2(r_[1], r_[0]) / d2r

        return (newlon.reshape(shape), newlat.reshape(shape))


    def local2geo(self, lon, lat, reverse = False):
        """ Convert from local coordinates to geocentric coordinates 

        lon and lat must have the same shape. Shapes are preserved in output

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]
        reverse: bool, optional
            set to False (default) if you want to rate from local to geocentric, 
            set to True if you want the opposite rotation

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system

        Note
        ----
        See self.geo2local for implementation
        """
        if reverse:
            return self.geo2local(lon, lat)
        else:
            return self.geo2local(lon, lat, reverse = True)


    def local2geo_enu_rotation(self, lon, lat):
        """ Calculate rotation matrices that transform local ENU to geocentric ENU

        Parameters
        ----------
        lon: array-like
            array of longitudes (local coords) for which rotation matrices should be calculated
        lat: array-like
            array of latitudes (local coords) for which rotation matrices should be calculated

        Returns
        -------
        R_localenu2geoenu: array
            Rotation matrices that rotate ENU vectors in local coordinates to ENU vectors
            in geocentric coordinates. Shape is (N, 2, 2). To get the opposite rotation, 
            use the transpose by swapping the last two axes of the array. The rotation 
            matrices are (2, 2), and should be applied on (east, north) components. The 
            upward component is the same in the two coordinate systems. 
            N is the size of lon and lat (they will be flattened)
        """

        th = (90 - np.array(lat).flatten()) * d2r
        ph = np.array(lon).flatten() * d2r

        # from ENU to ECEF:
        e_R = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
        n_R = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
        u_R = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

        R_enulocal2eceflocal = np.stack((e_R, n_R, u_R), axis = 2) # (N, 3, 3) with e n u in columns

        # from local to geocentric:
        lon_G, lat_G = self.local2geo(lon, lat)
        th = (90 - lat_G) * d2r
        ph = lon_G * d2r

        e_G = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
        n_G = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
        u_G = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

        R_ecefgeo2enugeo = np.stack((e_G, n_G, u_G), axis = 1) # (N, 3, 3) with e n u in rows

        # Combine:
        R_enulocal2ecefgeo = np.einsum('ij , njk -> nik', self.R_local2geo, R_enulocal2eceflocal)
        R_enulocal2enugeo  = np.einsum('nij, njk -> nik', R_ecefgeo2enugeo, R_enulocal2ecefgeo)

        # the result should describe a 2D rotation matrix:
        assert np.all( np.isclose(R_enulocal2enugeo[:, 2, 2], 1 ))
        assert np.all( np.isclose(R_enulocal2enugeo[:, 2, np.array([0, 1])], 0 ))
        assert np.all( np.isclose(R_enulocal2enugeo[:, np.array([0, 1]), 2], 0 ))
        return R_enulocal2enugeo[:, :2, :2] # (N, 2, 2)


    def vector_cube_projection(self, east, north, lon, lat):
        """ Calculate vector components projected on cube
        
        Perfor vector rotation from geographic system to cube
        system, using self.local2geo_enu_rotation and equation
        (14) of Ronchi et al. 

        Parameters
        ----------
        east: array-like
            Array of N eastward (geo) components
        north: array-like
            Array of N northward (geo) components
        lon: array-like
            Array of N longitudes that represent vector positions
        lat: array-like
            Array of N latitudes that represent vector positions

        Returns
        -------
        Axi: array-like
            N element array of vector components in xi direction
        Aeta: array-like
            N element array of vector components in eta direction

        """

        east, north, lon, lat = tuple(map(lambda x: np.array(x).flatten(), 
                                          [east, north, lon, lat]))
        Ageo = np.vstack((east, north)).T

        # rotation from geo to local:
        local_lon, local_lat = self.geo2local(lon, lat)
        R_enu_global2local = self.local2geo_enu_rotation(local_lon, local_lat)
        Alocal = np.einsum('nji, nj->ni', R_enu_global2local, Ageo).T

        # rearrange to south, east instead of east, north:
        Alocal = np.vstack((-Alocal[1], Alocal[0])).T

        # calculate the parameters used in transformation matrix:
        xi, eta = self.geo2cube(lon, lat)
        X   = np.tan(-xi)
        Y   = np.tan(-eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)
        dd = np.sqrt(delta - 1)

        # calculate transformation matrix elements:
        R = np.empty((east.size, 2, 2))
        R[:, 0, 0] = -D * X / dd 
        R[:, 0, 1] =  D * Y / dd / np.sqrt(delta)
        R[:, 1, 0] = -C * Y / dd
        R[:, 1, 1] = -C * X / dd / np.sqrt(delta)

        # rotate and return
        Acube = np.einsum('nij, nj->ni', R, Alocal).T

        # components in xi and eta directions:
        return Acube[0], Acube[1]




    def get_projected_coastlines(self, **kwargs):
        """ generate coastlines in projected coordinates """

        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = '50m'
        if 'category' not in kwargs.keys():
            kwargs['category'] = 'physical'
        if 'name' not in kwargs.keys():
            kwargs['name'] = 'coastline'

        shpfilename = shpreader.natural_earth(**kwargs)
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
        for coastline in coastlines:
            for line in coastline.geometry.geoms:
                lon, lat = np.array(line.coords[:]).T 
                yield self.geo2cube(lon, lat)

    def differentials(self, xi, eta, dxi, deta, R = 1):
        """ calculate magnitudes of line and surface elements 

        Implementation of equations 18-20 of Ronchi et al. 

        Broadcasting rules apply, so that output will have the shape of
        the combination of input parameters:
        dS.shape will be equal to (xi * eta * dxi * deta).shape

        xi, eta, dxi, deta must all be given in radians. dlxi and dleta
        will be given in units of R, and dS in units of R squared (default
        is radian and steradian)

        Parameters
        ----------
        xi: array-like
            xi coordinate(s) of surface element(s)
        eta: array-like
            eta coordinate(s) of surface element(s)
        dxi: array-like
            dimension(s) of surface element(s) in xi direction
        deta: array-like
            dimension(s) of surface element(s) in eta direction
        R: float, optional
            radius of the sphere - default is 1

        Returns
        -------
        dlxi: array-like
            Length of line element(s), in radians or in unit of R,
            along xi direction
        dleta: array-like
            Length of line element(s), in radians or in unit of R,
            along eta direction
        dS: array-like
            Area(s) of surface element(s), in steradians or in 
            the unit of R squared
        """

        X = np.tan(xi)
        Y = np.tan(eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)

        dlxi  = R * D * dxi  / (delta * np.cos( xi)**2)
        dleta = R * C * deta / (delta * np.cos(eta)**2)

        dS = R**2 * deta * dxi / (delta**(3./2) * np.cos(xi)**2 * np.cos(eta)**2)

        return dlxi, dleta, dS






class CSgrid(object):
    def __init__(self, projection, L, W, Lres, Wres, wshift = 0, R = 6371.2):
        """ set up grid for cubed sphere projection 
        
        Create a regular grid in xi,eta-coordinates. The grid will cover a 
        region of the cube surface that is L by W, where L is the dimension along
        the projection.orientation vector. The center of the grid is located at
        projection.position. 

        Parameters
        ----------
        projection: CSprojection
            CSprojection
        L: float
            Dimension of grid along CSprojection.orientation, i.e. the "length"
            of the grid. Dimension corresponds to the dimension of R at the 
            cube-sphere intersection point
        W: float
            Dimension of grid perpendicular CSprojection.orientation, i.e. the 
            "width" of the grid. Dimension corresponds to the dimension of R at 
            the cube-sphere intersection point 
        Lres: float or int
            If float, Lres denotes the size of grid cells in L direction, with 
            dimension same as R (at cube-sphere intersection point)
            if int, Lres denotes the number of grid cells in the Lres direction
        Wres: float or int
            If Lres is float, Wres denotes the size of grid cells in W direction, with 
            dimension same as R (at cube-sphere intersection point). If Lres is int, 
            Wres denotes the number of grid cells in the Wres direction
        wshift: float, optional
            Distance, in units of R, by which to move the grid in the xi-direction, 
            or W direction. Positive numbers will move the center right (towards
            positive xi)
        R: float (optional)
            Radius of the sphere. Default is 6371.2 (~Earth's radius in km)

        """
        self.projection = projection
        self.R = R
        self.wshift = wshift

        # normalize L and H to unit square:
        self.L = L / self.R
        self.W = W / self.R

        # make xi and eta arrays for the grid cell boundaries:
        if isinstance(Lres, int):
            xi_edge  = np.linspace(-np.arctan(W/R)/2, np.arctan(W/R)/2, Lres + 1) - wshift/self.R
            eta_edge = np.linspace(-np.arctan(L/R)/2, np.arctan(L/R)/2, Wres + 1)
        else:
            xi_edge  = np.r_[-np.arctan(W/R)/2:np.arctan(W/R)/2:np.arctan(Wres/(R))] - wshift/self.R
            eta_edge = np.r_[-np.arctan(L/R)/2:np.arctan(L/R)/2:np.arctan(Lres/(R))]

        # outer grid limits in xi and eta coords:
        self.xi_min, self.xi_max = xi_edge.min(), xi_edge.max()
        self.eta_min, self.eta_max = eta_edge.min(), eta_edge.max()

        # number of grid cells in L (eta) and W (xi) directions:
        self.NL, self.NW = len(eta_edge) - 1, len(xi_edge) - 1

        # size of grid cells in xi, eta coordinates:
        self.dxi  = xi_edge [1] - xi_edge [0]
        self.deta = eta_edge[1] - eta_edge[0]
        
        # xi, eta coordinates of cell corners:
        self.xi_mesh, self.eta_mesh = np.meshgrid(xi_edge, eta_edge, indexing = 'xy')

        # lon, lat coordiantes of cell corners:
        self.lon_mesh, self.lat_mesh = self.projection.cube2geo(self.xi_mesh, self.eta_mesh)

        # xi, eta coordinates of grid points (cell centers):
        self.xi  = self.xi_mesh [0:-1, 0:-1] + self.dxi  / 2
        self.eta = self.eta_mesh[0:-1, 0:-1] + self.deta / 2

        # geocentric lon, lat [deg] of grid points:
        self.lon, self.lat = self.projection.cube2geo(self.xi, self.eta)
        self.local_lon, self.local_lat = self.projection.geo2local(self.lon, self.lat)

        # geocentric lon, colat [rad] of grid points:
        self.phi, self.theta = self.lon * d2r, (90 - self.lat) * d2r

        # longitude and colatitude of grid points in local spherical coords:
        self.local_phi, self.local_theta = self.local_lon * d2r, (90 - self.local_lat) * d2r

        # cubed square parameters for grid points (cell centers)
        self.X = np.tan(-self.xi)
        self.Y = np.tan(-self.eta)
        self.delta = 1 + self.X**2 + self.Y**2
        self.C = np.sqrt(1 + self.X**2)
        self.D = np.sqrt(1 + self.Y**2)



    def _index(self, i, j):
        """ Calculate the 1D index that corresponds to the grid index i, j

        Parameters
        ----------
        i: array-like (int)
            row index(es)
        j: array-like (int)
            columns index(es)

        Returns
        -------
        1D array of ints which denote the index(es) of i, j in a flattened version
        of a 2D array of shape (self.NL, self.NW)
        """
        i = np.array(i)
        j = np.array(j)

        # handle negative indices:
        i[i < 0] = self.NL + i[i < 0]
        j[j < 0] = self.NW + j[j < 0]
        
        try:
            return np.ravel_multi_index((i, j), (self.NL, self.NW))
        except:
            print('invalid index?', i, j, self.NL, self.NW)


    def get_grid_boundaries(self, geocentric = True):
        """ get grid boundaries for plotting 
            
            yields tuples of (lon, lat) arrays that outline
            the grid cell boundaries. 

            Example:
            --------
            for c in obj.get_grid_boundaries():
                lon, lat = c
                plot(lon, lat, 'k-', transform = ccrs.Geocentric())
        """
        if geocentric:
            x, y = self.lon_mesh, self.lat_mesh
        else:
            x, y = self.xi_mesh , self.eta_mesh

        for i in range(self.NL + self.NW + 2):
            if i < self.NL + 1:
                yield (x[i, :], y[i, :])
            else:
                i = i - self.NL - 1
                yield (x[:, i], y[:, i])


    def get_Le_Ln(self, order = None, return_dxi_deta = False):
        """ calculate the matrix that produces the derivative of an object in self.field
            in the eastward direction

        Not implemented/TODO: order
        """

        dxi = self.dxi
        det = self.deta
        N = self.NL
        M = self.NW

        D_xi = np.zeros((N * M, N * M))
        D_et = np.zeros((N * M, N * M))

        i , j  = np.arange(N), np.arange(M)

        # inner cells:
        ii, jj = np.meshgrid(i[1:-1], j[1:-1])
        D_xi[self._index(ii, jj), self._index(ii    , jj + 1)] += 1. / (2 * dxi)
        D_xi[self._index(ii, jj), self._index(ii    , jj - 1)] -= 1. / (2 * dxi)
        D_et[self._index(ii, jj), self._index(ii + 1, jj    )] += 1. / (2 * det)
        D_et[self._index(ii, jj), self._index(ii - 1, jj    )] -= 1. / (2 * det)

        # edges, derivative wrt to xi
        D_xi[self._index( 0,  j[1:-1]), self._index( 0,j[1:-1] + 1)] += 1. / (2 * dxi)
        D_xi[self._index( 0,  j[1:-1]), self._index( 0,j[1:-1] - 1)] -= 1. / (2 * dxi)
        D_xi[self._index(-1,  j[1:-1]), self._index(-1,j[1:-1] + 1)] += 1. / (2 * dxi)
        D_xi[self._index(-1,  j[1:-1]), self._index(-1,j[1:-1] - 1)] -= 1. / (2 * dxi)
        D_xi[self._index(i,  0), :] = D_xi[self._index(i,  1), :] # TODO: improve edges
        D_xi[self._index(i, -1), :] = D_xi[self._index(i, -2), :] 

        # edges, derivative wrt to eta
        D_et[self._index(i[1:-1],  0), self._index(i[1:-1] + 1,  0)] += 1. / (2 * det)
        D_et[self._index(i[1:-1],  0), self._index(i[1:-1] - 1,  0)] -= 1. / (2 * det)
        D_et[self._index(i[1:-1], -1), self._index(i[1:-1] + 1, -1)] += 1. / (2 * det)
        D_et[self._index(i[1:-1], -1), self._index(i[1:-1] - 1, -1)] -= 1. / (2 * det)
        D_et[self._index( 0, j)] = D_et[self._index( 1, j)] # TODO: imrove edges
        D_et[self._index(-1, j)] = D_et[self._index(-2, j)] 

        if return_dxi_deta:
            return D_xi, D_et

        # convert to gradient compnents
        X = self.X.flatten().reshape((1, -1))
        Y = self.Y.flatten().reshape((1, -1))
        D = self.D.flatten().reshape((1, -1))
        C = self.C.flatten().reshape((1, -1))
        d = self.delta.flatten().reshape((1, -1))

        # equation 21 of Ronchi et al.
        L_xi = (D     * D_xi     + X * Y * D_et / D) / self.R
        L_et = (X * Y * D_xi / C +     C * D_et    ) / self.R
        dd = np.sqrt(d - 1)

        # conversion from xi/eta to geocentric east/west is accomplished through the
        # matrix in equation 14 of Ronchi et al. (instead of 12 since the signs of 
        # xi/eta are reversed in this implementation) 
        # The elements of this matrix are:
        a00 = -D * X / dd 
        a01 =  D * Y / dd / np.sqrt(d)
        a10 = -C * Y / dd
        a11 = -C * X / dd / np.sqrt(d)        

        # The a matrix converts from local theta/phi to xi/eta. The elements of
        # the inverse are:
        det = a00*a11 - a01*a10
        b00 =  a11 /det 
        b01 = -a01 /det 
        b10 = -a10 /det 
        b11 =  a11 /det 

        # Combine this with the rotation matrix from local east/north to geocentric east/south:
        lon, lat = self.projection.geo2local(self.lon.flatten(), self.lat.flatten())
        R = self.projection.local2geo_enu_rotation(lon, lat)
        r10 =  R[:, 0, 0].reshape((1, -1))
        r11 =  R[:, 0, 1].reshape((1, -1))
        r00 =  R[:, 1, 0].reshape((1, -1))
        r01 =  R[:, 1, 1].reshape((1, -1))
        # where I have switched the order of the rows and multiplied first row by -1
        # so that R acts on (south/east) instead of (east/north). 
        # This is consistent with b, so we can combine the operations:
        x00 = r00*b00 + r01*b10
        x01 = r00*b01 + r01*b11
        x10 = r10*b00 + r11*b10
        x11 = r10*b01 + r11*b11

        # finally the matrices that calculate the east/north components
        L_e = x00 * L_xi + x01 * L_et
        L_s = x10 * L_xi + x11 * L_et

        return L_e, -L_s

