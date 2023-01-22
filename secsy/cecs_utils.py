"""CECS utils

NOTE: Here we use a Cartesian coordinate system x,y,z such that z is UPWARD
(and, say, x is east and y is north). This is DIFFERENT from the Cartesian
coordinate system that Vanhamäki (2007) and Vanhamäki, Viljanen, and Amm (2005)
use to define CECS. For them, x is north, y is east, and z is DOWN.

Why does it matter? Because if you compare expressions for J and B in this
implementation with their expressions, you will notice sign differences.

Also, some conventions (established by references, not me):
•A positive-amplitude CECS CF system corresponds to a DOWNWARD field-aligned
 current and a radially OUTWARD current sheet.
•A positive-amplitude CECS DF system corresponds to a current that circulates in
 the CLOCKWISE direction when seen from above (i.e., the -phihat direction).


REFERENCES
==========

Vanhamäki, H. (2007) ‘Theoretical modeling of ionospheric electrodynamics
including induction effects’, Finnish Meteorological Institute Contributions.

Vanhamäki, H., Viljanen, A. and Amm, O. (2005) ‘Induction effects on ionospheric
electric and magnetic fields’, Annales Geophysicae, 23(5), pp. 1735–1746. doi:
10.5194/angeo-23-1735-2005.

Spencer M. Hatch
October 2022
"""

import numpy as np
d2r = np.pi/180
MU0 = 4 * np.pi * 1e-7


def get_prime_coordinates(x,y,z, x_cecs, y_cecs, z_cecs):
    
    x   = np.array(x).flatten()[:, np.newaxis]
    y   = np.array(y).flatten()[:, np.newaxis]
    z   = np.array(z).flatten()[:, np.newaxis]
    x_c = np.array(x_cecs).flatten()[np.newaxis, :]
    y_c = np.array(y_cecs).flatten()[np.newaxis, :]
    z_c = np.array(z_cecs).flatten()[np.newaxis, :]

    xp = x-x_c
    yp = y-y_c
    zp = z-z_c

    return xp,yp,zp


def get_distance(x, y, z, x_cecs, y_cecs, z_cecs):
    """" calculate distance between data point and cecs node.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    return_degrees: bool, optional
        Set to True if you want output in degrees. Default is False (radians)

    Returns
    -------
    dist: 2D array (x.size, x_cecs.size)
        Array of distances between the points
        described by (x, y, z) and the points described by 
        (x_cecs, y_cecs, z_cecs).
    """

    # reshape
    xp, yp, zp = get_prime_coordinates(x,y,z,x_cecs,y_cecs,z_cecs)

    dist = np.sqrt(xp**2+yp**2+zp**2)

    return dist


def get_rho(x, y, z, x_cecs, y_cecs, z_cecs):
    """" calculate rho (horizontal distance) between data points and cecs nodes.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size

    Returns
    -------
    rho: 2D array (x.size, x_cecs.size)
        Array of horizontal distances between data points
        described by (x, y, z) and cecs nodes at (x_cecs, y_cecs, z_cecs).
    """

    xp, yp, zp = get_prime_coordinates(x,y,z,x_cecs,y_cecs,z_cecs)

    rho = np.sqrt( xp**2+yp**2)

    return rho


def get_phi(x, y, z, x_cecs, y_cecs, z_cecs, return_degrees = False):
    """" calculate polar (horizontal) angle of data point relative to cecs node.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    return_degrees: bool, optional
        Set to True if you want output in degrees. Default is False (radians)

    Returns
    -------
    phi: 2D array (x.size, x_cecs.size)
        Array of polar (horizontal) angles of data points
        described by (x, y, z) relative the points described by 
        (x_cecs, y_cecs, z_cecs). Unit in radians unless return_degrees is set
        to True
    """

    xp, yp, zp = get_prime_coordinates(x,y,z,x_cecs,y_cecs,z_cecs)

    phi = np.arctan2(yp,xp)

    if return_degrees:
        phi = phi/d2r

    return phi


def get_phihat(x,y,z, x_cecs, y_cecs, z_cecs):
    """" calculate polar (horizontal) unit vector for data point relative to cecs node.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size

    Returns
    -------
    phihat: 3D array (x.size, x_cecs.size, 3)
        Array of polar (horizontal) unit vectors in Cartesian coordinates of data points
        described by (x, y, z) relative the points described by 
        (x_cecs, y_cecs, z_cecs).
    """
    
    phiprime = get_phi(x, y, z, x_cecs, y_cecs, z_cecs)
    
    phihat = np.transpose(np.stack([-np.sin(phiprime),np.cos(phiprime),np.zeros_like(phiprime)]),axes=[1,2,0])

    return phihat


def get_rhohat(x,y,z, x_cecs, y_cecs, z_cecs):
    """" calculate rho unit vector from cecs node to data point.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    lon_cecs: array-like
        Array of CECS pole longitudes [deg]
        Flattened array must havef same size as lat_cecs
        Output will be a 2D array with shape (mlat.size, mlat_cecs.size)

    Returns
    -------
    rhohat: 3D array (3, x.size, x_cecs.size)
        Array of rho unit vectors in Cartesian coordinates pointing from
        cecs nodes to data points.
    """
    
    phiprime = get_phi(x, y, z, x_cecs, y_cecs, z_cecs)
    
    rhohat = np.transpose(np.stack([np.cos(phiprime),np.sin(phiprime),np.zeros_like(phiprime)]),axes=[1,2,0])

    return rhohat


def get_CECS_J_G_matrices(x,y,z, x_cecs, y_cecs, z_cecs, 
                          current_type = 'divergence_free', constant = 1./(2*np.pi), 
                          dz_tolerance=10.,
):
    """ Calculate matrices Ge and Gn which relate CECS amplitudes to current density 
        vector components.

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    current_type: string, optional
        The type of CECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
        'potential': scalar field whose negative gradient is curl-free CECS
        'scalar': 
    constant: float, optional
        The CECS functions are scaled by the factor 1/(2pi), which is
        the default value of 'constant'. Change if you want something 
        different.
    dz_tolerance: Since CECS were not originally designed to be used in 3D, we have to make sure that
                      in evaluating the current at a particular altitude, the CECS poles at higher or lower
                      altitudes (within tolerance given by dz_tolerance) do not contribute to the
                      evaluated current

    Returns
    -------
    If current_type is 'divergence_free' or 'curl_free':
    Gx: 2D array
        2D array with shape (x.size, x_cecs.size), relating CECS amplitudes
        m to the x-direction current densities at (x, y, z) via 'jx = Gx.dot(m)'
    Gy: 2D array
        2D array with shape (x.size, y_cecs.size), relating CECS amplitudes
        m to the y-direction current densities at (x, y, z) via 'jy = Gy.dot(m)'
    # Gz: 2D array
    #     2D array with shape (x.size, y_cecs.size), relating CECS amplitudes
    #     m to the z-direction current densities at (x, y, z) via 'jz = Gz.dot(m)'
    If current_type is 'potential' or 'scalar':
    G: 2D array
        2D array with shape (lat.size, lat_cecs.size), relating amplitudes m
        to scalar field magnitude at (lat, lon) via 'z = G.dot(m)'

    """
    
    if current_type not in ['divergence_free','curl_free']:
        assert 2<0,"types 'potential' and 'scalar' not yet implemented!"

    xp, yp, zp = get_prime_coordinates(x,y,z,x_cecs,y_cecs,z_cecs)

    rho = get_rho(x,y,z,x_cecs,y_cecs,z_cecs)

    if current_type == 'divergence_free':
        unit_vec = (-1.)*get_phihat(x,y,z,x_cecs,y_cecs,z_cecs)
    elif current_type == 'curl_free':
        unit_vec = get_rhohat(x,y,z,x_cecs,y_cecs,z_cecs)
    elif current_type in ['potential', 'scalar']:
        unit_vec = 1
    else:
        raise Exception('type must be "divergence_free", "curl_free", "potential", or "scalar"')

    # get the scalar part of Amm's divergence-free CECS:    
    if current_type in ['divergence_free', 'curl_free']:
        coeff = constant / rho

        # G matrices
        Gx = coeff * unit_vec[:, :, 0]
        Gy = coeff * unit_vec[:, :, 1]
    
        shouldbezero = np.abs(zp) > dz_tolerance
        Gx[shouldbezero] = 0.
        Gy[shouldbezero] = 0.

        return Gx, Gy
    else: # current_type is 'potential' or 'scalar'
        # NOT SURE WHAT THESE SHOULD BE
        # if current_type == 'potential':
        #     return -2*constant*np.log(np.sin(theta/2)).T
        # elif current_type == 'scalar':
        #     return    constant      / np.tan(theta/2).T

        return np.nan, np.nan


def get_CECS_B_G_matrices(x, y, z, x_cecs, y_cecs, z_cecs,
                          current_type = 'divergence_free',
                          constant = MU0/(4.*np.pi), 
):
    """ Calculate matrices Gx, Gy, and Gz that relate CECS amplitudes to magnetic field

    Based on equations (A.10) and (A.11) of Vanhamäki (2007).

    Parameters
    ----------
    x,y,z: array-like
        Array of Cartesian coordinate of evaluation points [m]
        Flattened arrays must all have the same size
    {x,y,z}_cecs: array-like
        Array of CECS pole Cartesian coordinates [m]
        Flattened arrays must all have the same size
    current_type: string, optional
        The type of CECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
    constant: float, optional
        The B^cf CECS function is scaled by the factor MU0/(2pi), while
        B^df CECS function is scaled by MU0/(4pi)

    Returns
    -------
    Gx: 2D array
        2D array with shape (x.size, x_cecs.size), relating CECS amplitudes
        m to the x-component magnetic field at (x, y, z) via 'Bx = Gx.dot(m)'
    Gy: 2D array
        2D array with shape (y.size, y_cecs.size), relating CECS amplitudes
        m to the y-component magnetic field at (x, y, z) via 'By = Gy.dot(m)'
    Gz: 2D array
        2D array with shape (z.size, z_cecs.size), relating CECS amplitudes
        m to the z-component magnetic field at (x, y, z) via 'Bz = Gz.dot(m)'


    Notes
    ------
    Variables with 'p' in name signify a quantity in the CECS node's local (i.e., "p"rimed) coordinate system

    2022/10/15 SMH
    """
    
    # reshape z:
    if np.array(z).size == 1:
        z = np.ones_like(x) * z
    else:
        z = np.array(z).flatten()[:, np.newaxis]

    xp, yp, zp = get_prime_coordinates(x,y,z,x_cecs,y_cecs,z_cecs)

    rho = get_rho(x,y,z,x_cecs,y_cecs,z_cecs)

    coeff = constant / rho

    # G matrix scale factors
    if current_type == 'divergence_free':

        rhohat = get_rhohat(x,y,z,x_cecs,y_cecs,z_cecs)
        zhat   = np.zeros_like(rhohat)
        zhat[:,:,2] = 1.

        Grhoprime = -coeff * (1 - np.abs(zp) / np.sqrt(rho**2+zp**2)) * np.sign(zp)

        Gx = Grhoprime * rhohat[:,:,0]
        Gy = Grhoprime * rhohat[:,:,1]

        Gz = -coeff * rho/np.sqrt(rho**2+zp**2)


    elif current_type == 'curl_free':

        phihat = get_phihat(x,y,z,x_cecs,y_cecs,z_cecs)

        heaviside = np.zeros_like(zp,dtype=zp.dtype)
        heaviside[((zp) > -1e8) | (np.isclose(0.,zp))] = 1.

        Gphiprime = -2. * coeff * heaviside

        Gx = Gphiprime * phihat[:,:,0]
        Gy = Gphiprime * phihat[:,:,1]
        Gz = np.zeros_like(phihat[:,:,2])

    return Gx, Gy, Gz


if __name__ == '__main__':
    import hatch_python_utils.math.cecs_utils
    importlib.reload(hatch_python_utils.math.cecs_utils)
    from hatch_python_utils.math import cecs_utils
    import importlib 
    importlib.reload(cecs_utils)
    print("TEST OUT CECS")
    # x,y,z = np.array([[1,0,0],[2,0,0],[3,0,0]]).T
    # xc,yc,zc = np.array([[0,0,-1],[0,0,0],[0,0,1]]).T
    xyz = np.array([[1,0,0],[1,1,0],[-1,1,1]]).T
    xyzc = np.array([[0,0,-1],[0,0,0],[0,0,1],[1,0,1],[-5,0,1]]).T
    # cecs_utils.get_phi(x,y,z,xc,yc,zc)
    phi = cecs_utils.get_phi(*xyz,*xyzc,return_degrees=True)
    phihat = cecs_utils.get_phihat(*xyz,*xyzc)
    rhohat = cecs_utils.get_rhohat(*xyz,*xyzc)
    Gx, Gy = cecs_utils.get_CECS_J_G_matrices(*xyz,*xyzc)

    GBx, GBy, GBz = cecs_utils.get_CECS_B_G_matrices(*xyz,*xyzc,current_type='divergence_free')

    print(phi.shape)
    print(phi)


