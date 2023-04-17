""" 
SECS utils

"""

import numpy as np
from .spherical import enu_to_ecef
d2r = np.pi/180
MU0 = 4 * np.pi * 1e-7
RE = 6371.2 * 1e3


def dpclip(x, delta = 1e-7):
    """ 
    dot product clip:
    clip x to values between -1 + delta and 1 - delta
    """
    return np.clip(x, -1 + delta, 1 - delta)

def get_theta(lat, lon, lat_secs, lon_secs, return_degrees = False):
    """" calculate theta angle - the angle between data point and secs node.

    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
        Output will be a 2D array with shape (mlat.size, mlat_secs.size)
    return_degrees: bool, optional
        Set to True if you want output in degrees. Default is False (radians)

    Returns
    -------
    theta: 2D array (lat.size, lat_secs.size)
        Array of polar angles, angular distances between the points
        described by (lat, lon) and the points described by 
        (lat_secs, lon_secs). Unit in radians unless return_degrees is set
        to True
    """

    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T

    # the polar angles (N, M):
    theta = np.arccos(dpclip(np.einsum('ij, kj -> ik', ecef_r_data, ecef_r_secs)))

    if return_degrees:
        theta = theta / d2r 

    return theta


def get_SECS_J_G_matrices(lat, lon, lat_secs, lon_secs, 
                          current_type = 'divergence_free', constant = 1./(4*np.pi), 
                          RI = RE + 110 * 1e3,
                          singularity_limit = 0):
    """ Calculate matrices Ge and Gn which relate SECS amplitudes to current density 
        vector components.

    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
    current_type: string, optional
        The type of SECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
        'potential': scalar field whose negative gradient is curl-free SECS
        'scalar': 
    constant: float, optional
        The SECS functions are scaled by the factor 1/(4pi), which is
        the default value of 'constant'. Change if you want something 
        different.
    RI: float (optional)
        Radius of SECS poles. Default is Earth radius + 110,000 m
    singularity_limit: float (optional)
        A modified version of the SECS functions will be used at
        points that are closer than singularity_limit. The modification
        is given by equations 2.43 (CF) and 2.44 (DF) in Vanhamaki and
        Juusola (2020), and singularity_limit / RI is equal to theta0
        in these equations. Default is 0, which means that the original
        version of the SECS functions are used (with singularities). 
        singularity_limit is ignored if current_type is 'potential' or 'scalar'

    Returns
    -------
    If current_type is 'divergence_free' or 'curl_free':
    Ge: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the eastward current densities at (lat, lon) via 'je = Ge.dot(m)'
    Gn: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the northward current densities at (lat, lon) via 'jn = Gn.dot(m)'
    If current_type is 'potential' or 'scalar':
    G: 2D array
        2D array with shape (lat.size, lat_secs.size), relating amplitudes m
        to scalar field magnitude at (lat, lon) via 'z = G.dot(m)'

    """
    
    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T
    
    # unit vector pointing from SECS to data points - (M, N, 3) 
    ecef_t = ecef_r_secs[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik->ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECS:
    R = np.hstack( (np.dstack((-np.sin(lo)              ,  np.cos(lo)             , np.zeros_like(la) )),
                    np.dstack((-np.cos(lo)  * np.sin(la), -np.sin(lo) * np.sin(la), np.cos(       la) )),
                    np.dstack(( np.cos(lo)  * np.cos(la),  np.sin(lo) * np.cos(la), np.sin(       la) ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECS
    enu_t = np.einsum('lij, lkj->lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion
    
    if current_type == 'divergence_free':
        # rotate these vectors to get vectors pointing eastward with respect to SECS systems at each data point
        enu_vec = np.dstack((enu_t[:, :, 1], -enu_t[:, :, 0])) # north -> east and east -> south
    elif current_type == 'curl_free':
        enu_vec = -enu_t # outward from SECS
    elif current_type in ['potential', 'scalar']:
        enu_vec = 1
    else:
        raise Exception('type must be "divergence_free", "curl_free", "potential", or "sclar"')

    # get the scalar part of Amm's divergence-free SECS:    
    theta  = np.arccos(dpclip(np.einsum('ij,kj->ik', ecef_r_secs, ecef_r_data)))
    if current_type in ['divergence_free', 'curl_free']:
        coeff = constant /np.tan(theta/2)/ RI

        # apply modifications to handle singularities:
        theta0 = singularity_limit / RI
        if theta0 > 0:
            alpha = 1 / np.tan(theta0/2)**2
            coeff[theta < theta0] = constant * alpha * np.tan(theta[theta < theta0]/2) / RI

        # G matrices
        Ge = coeff * enu_vec[:, :, 0].T
        Gn = coeff * enu_vec[:, :, 1].T
    
        return Ge.T, Gn.T
    else: # current_type is 'potential' or 'scalar'
        if current_type == 'potential':
            return -2*constant*np.log(np.sin(theta/2)).T
        elif current_type == 'scalar':
            return    constant      / np.tan(theta/2).T


def get_Nakano_cf_G_matrices(lat, lon, lat_secs, lon_secs, 
                             current_type = 'curl_free', 
                             constant = 1./(4*np.pi), 
                             eta = 131.4,
                             RI = RE + 110 * 1e3):
    """ Calculate matrices Ge and Gn which relate SECS amplitudes to current density 
        vector components.

    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
    current_type: string, optional
        The type of SECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
        'potential': scalar field whose negative gradient is curl-free SECS
        'scalar': 
    eta: float
        Analogous to 1/variance of a typical Gaussian distribution:
        Whereas the width of a Gaussian increases with increasing variance,
        the width of the spherical Gaussian _decreases_ with increasing eta.
        The default value used in Fig 1 of Nakano et al (2020) is 131.4.
    constant: float, optional
        The SECS functions are scaled by the factor 1/(4pi), which is
        the default value of 'constant'. Change if you want something 
        different.
    RI: float (optional)
        Radius of SECS poles. Default is Earth radius + 110,000 m

    Returns
    -------
    If current_type is 'divergence_free' or 'curl_free':
    Ge: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the eastward current densities at (lat, lon) via 'je = Ge.dot(m)'
    Gn: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the northward current densities at (lat, lon) via 'jn = Gn.dot(m)'
    If current_type is 'potential' or 'scalar':
    G: 2D array
        2D array with shape (lat.size, lat_secs.size), relating amplitudes m
        to scalar field magnitude at (lat, lon) via 'z = G.dot(m)'

    """
    
    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T
    
    # unit vector pointing from SECS to data points - (M, N, 3) 
    ecef_t = ecef_r_secs[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik->ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECS:
    R = np.hstack( (np.dstack((-np.sin(lo)              ,  np.cos(lo)             , np.zeros_like(la) )),
                    np.dstack((-np.cos(lo)  * np.sin(la), -np.sin(lo) * np.sin(la), np.cos(       la) )),
                    np.dstack(( np.cos(lo)  * np.cos(la),  np.sin(lo) * np.cos(la), np.sin(       la) ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECS
    enu_t = np.einsum('lij, lkj->lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion
    
    # if current_type == 'divergence_free':
    #     # rotate these vectors to get vectors pointing eastward with respect to SECS systems at each data point
    #     enu_vec = np.dstack((enu_t[:, :, 1], -enu_t[:, :, 0])) # north -> east and east -> south
    # elif current_type == 'curl_free':
    enu_vec = -enu_t # outward from SECS poles
    # elif current_type in ['potential', 'scalar']:
    #     enu_vec = 1
    # else:
    #     raise Exception('type must be "divergence_free", "curl_free", "potential", or "sclar"')

    # get the scalar part of Amm's divergence-free SECS:    
    costheta = dpclip(np.einsum('ij,kj->ik', ecef_r_secs, ecef_r_data))
    theta  = np.arccos(costheta)

    psi_cf = np.exp(eta * (costheta - 1) )
    if current_type == 'curl_free':
        # coeff = constant /np.tan(theta/2)/ RI

        # Dot product of ecef vectors
        # ecef_dot = np.sum(ecef_r_secs[np.newaxis, :, :] * ecef_r_data[:, np.newaxis, :], axis=2)
        # dtheta = np.arccos(ecef_dot)
        vcfmag = eta * np.sin(theta) * psi_cf

        coeff = constant * vcfmag / RI

        # # apply modifications to handle singularities:
        # theta0 = singularity_limit / RI
        # if theta0 > 0:
        #     alpha = 1 / np.tan(theta0/2)**2
        #     coeff[theta < theta0] = constant * alpha * np.tan(theta[theta < theta0]/2) / RI

        # G matrices
        Ge = coeff * enu_vec[:, :, 0].T
        Gn = coeff * enu_vec[:, :, 1].T
    
        return Ge.T, Gn.T
    else: # current_type is 'potential' or 'scalar'
        if current_type == 'potential':
            return constant * psi_cf.T 
        else:
            assert 2<0,f"The current type you've selected ('{current_type}') is invalid"
        # elif current_type == 'scalar':
        #     return    constant      / np.tan(theta/2).T
    

def get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs,
                          current_type = 'divergence_free', constant = 1./(4*np.pi), 
                          RI = RE + 110 * 1e3,
                          singularity_limit = 0,
                          induction_nullification_radius = None):
    """ Calculate matrices Ge, Gn, and Gr which relate SECS amplitudes to magnetic field

    Based on equations (9) and (10) of Amm and Viljanen 1999, or (2.13)-(2.14) in Vanhamaki
    and Juusola 2020. 

    If singularity_limit > 0, the magnetic field of curl-free currents is modified, but
    not the magnetic field of divergence-free currents (!). See Section 2.10.2 and 
    equation (2.46) in Vanhamaki and Juusola 2020.


    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    r: array-like
        Array of radii of evaluation points. Flattened
        array must either have size 1, in which case one
        radius is used for all points, or have same size as 
        lat. Unit should be the same as RI 
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
    current_type: string, optional
        The type of SECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
    constant: float, optional
        The SECS functions are scaled by the factor 1/(4pi), which is
        the default value of 'constant'. Change if you want something 
        different.
    RI: float (optional)
        Radius of SECS poles. Default is Earth radius + 110,000 m
    singularity_limit: float (optional)
        A modified version of the SECS functions will be used at
        points that are closer than singularity_limit. The modification
        is given by equations 2.43 (CF) and 2.44 (DF) in Vanhamaki and
        Juusola (2020), and singularity_limit / RI is equal to theta0
        in these equations. Default is 0, which means that the original
        version of the SECS functions are used (with singularities). 
    induction_nullification_radius: float or None, optional
        The radius at which ground induced image currents cancel the radial
        magnetic field. Default in None, in which case there are no
        induced image currents. This part is based on equations of Appendix A
        in "Juusola, L., Kauristie, K., Vanhamäki, H., Aikio, A., and 
        van de Kamp, M. (2016), Comparison of auroral ionospheric and field‐
        aligned currents derived from Swarm and ground magnetic field measurements, 
        J. Geophys. Res. Space Physics, 121, 9256– 9283, doi:10.1002/2016JA022961."

    Returns
    -------
    Ge: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the eastward magnetic field at (lat, lon) via 'Be = Ge.dot(m)'
    Gn: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the northward magnetic field at (lat, lon) via 'Bn = Gn.dot(m)'
    Gr: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the radial magnetic field at (lat, lon) via 'Br = Gr.dot(m)'

    """
    
    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # reshape r:
    if np.array(r).size == 1:
        r = np.ones_like(la) * r
    else:
        r = np.array(r).flatten()[:, np.newaxis]

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T
    
    # unit vector pointing from SECS to data points - (M, N, 3) 
    ecef_t = ecef_r_secs[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik->ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    np.seterr(invalid='ignore', divide='ignore')
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECS:
    R = np.hstack( (np.dstack((-np.sin(lo)              ,  np.cos(lo)             , np.zeros_like(la) )),
                    np.dstack((-np.cos(lo)  * np.sin(la), -np.sin(lo) * np.sin(la), np.cos(       la) )),
                    np.dstack(( np.cos(lo)  * np.cos(la),  np.sin(lo) * np.cos(la), np.sin(       la) ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECS
    enu_t = np.einsum('lij, lkj->lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion

    # the polar angles (N, M):
    theta = np.arccos(dpclip(np.einsum('ij, kj -> ik', ecef_r_data, ecef_r_secs)))


    # indices of data points that are below and above current sheet:
    below = r.flatten() <= RI
    above = r.flatten()  > RI

    # G matrix scale factors
    if current_type == 'divergence_free':
        s = np.minimum(r, RI) / np.maximum(r, RI)
        sa = s[above]
        sb = s[below]

        Ar  = MU0 * constant /  r                   # common factor radial direction
        Sr = np.zeros_like(theta)
        Sr[below] = 1  / np.sqrt(1 + sb**2 - 2 * sb * np.cos(theta[below])) - 1
        Sr[above] = sa / np.sqrt(1 + sa**2 - 2 * sa * np.cos(theta[above])) - sa
        Gr = Ar * Sr

        An_ = MU0 * constant / (r * np.sin(theta))  # common factor local northward (note sign difference wrt theta) direction
        Sn_ = np.zeros_like(theta)
        Sn_[below] = (sb -      np.cos(theta[below])) / np.sqrt(1 + sb**2 - 2 * sb * np.cos(theta[below])) + np.cos(theta[below])
        Sn_[above] = (1  - sa * np.cos(theta[above])) / np.sqrt(1 + sa**2 - 2 * sa * np.cos(theta[above])) - 1
        Gn_ = An_ * Sn_        

        # calculate geo east, north:
        Ge =  Gn_ * enu_t[:, :, 0]
        Gn =  Gn_ * enu_t[:, :, 1]

    elif current_type == 'curl_free':
        # G matrix for local eastward component
        Ge_ = -MU0 * constant / np.tan(theta/2) / r 

        # apply modifications to handle singularities:
        theta0 = singularity_limit / RI
        if theta0 > 0:
            alpha = 1 / np.tan(theta0/2)**2
            rr = np.tile(r, (1, theta.shape[1])) # need one r for every element in matrix
            Ge_[theta < theta0] = -MU0 * constant * alpha * np.tan(theta[theta < theta0]/2) / rr[theta < theta0]

        # zero below current sheet:
        Ge_[below] *= 0 

        # calculate geo east, north, radial:
        Ge =  Ge_ * enu_t[:, :, 1] # eastward component of enu_t is northward component of enu_e (unit vector in local east direction)
        Gn = -Ge_ * enu_t[:, :, 0] # northward component of enu_t is eastward component of enu_e
        Gr =  Ge_ * 0              # no radial component


    if induction_nullification_radius != None and current_type == 'divergence_free':
        # include the effect of telluric image currents
        radius = induction_nullification_radius**2 / RI
        amplitude_factor = -RI / induction_nullification_radius

        Ge_, Gn_, Gr_ = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, 
                                              current_type = 'divergence_free',
                                              RI = radius)
        Ge = Ge + amplitude_factor * Ge_
        Gn = Gn + amplitude_factor * Gn_
        Gr = Gr + amplitude_factor * Gr_



    return Ge, Gn, Gr


def get_wedge_G_matrix(lat, lon, r, lat_I, lon_I, r_I, Ie, In, Ir, calculate_radial_leg = True):
    """ Calculate matrices that map between the magnetic field components
        and the magnitudes of current wedges that consist of radial semi-infinite 
        line currents connected to another semi-infinite line current with 
        orientations and connection points specified in spherical coordinates

    This is useful for making a correction to the magnetic field of curl-free (CF)
    SECS for inclined magnetic field lines. The magnetic field of CF SECS with 
    inclined field lines can be expressed as the sum of the standard system with
    radial field lines and the magnetic field of a wedge calculated with this function. 

    Parameters
    ----------
    lat : array (size N or 1)
        latitudes of evaluation points [deg]
    lon : array (size N or 1)
        longitudes of evaluation points [deg]
    r : array (size N or 1)
        radii of evaluation points [m, unless other unit is consistently used]
    lat_I : array (size K or 1)
        latitudes of the wedge connection points [deg]
    lon_I : array (size K or 1)
        longitudes of the wedge connection points [deg]
    r_I : array (size K or 1)
        radii of of wedge connection points [same unit as r]
    Ie : array (size K or 1)
        eastward components of vectors along non-radial wedge legs
    In : array (size K or 1)
        eastward components of vectors along non-radial wedge legs
    Ir : array (size K or 1)
        radial components of vectors along non-radial wedge legs
    calculate_radial_leg : bool, default True
        used to determine if the magnetic field of a radial leg of the current
        wedge should be calculated in recursive call to the function. 

    Returns
    -------
    Ge : array (N X K)
        2D array that maps current magnitudes, I, to eastward magnetic fields, Be, such that
        Be = Ge.dot(I), where I is given in Ampere and Be in T
    Gn : array (N X K)
        2D array that maps current magnitudes, I, to northward magnetic fields, Bn, such that
        Be = Gn.dot(I), where I is given in Ampere and Bn in T
    Gr : array (N X K)
        2D array that maps current magnitudes, I, to radial magnetic fields, Br, such that
        Be = Gr.dot(I), where I is given in Ampere and Br in T

    Note
    ----
    The output of this function can be used in combination with get_SECS_B_G_matrices for curl-free
    currents to correct for the effect of inclined field lines. See get_CF_SECS_B_G_matrices_for_inclined_field().

    This is only a first-order correction, since it approximates the field line as an infinite straight line. 
    It should give an improvement at mid latitudes but it's not suitable at low latitudes. See work by 
    Heikki Vanhamäki for better ways to handle this. 

    Calculations are based on analytical expressions (see e.g., Fig 5.19 in Griffiths)
    """

    # turn input into arrays
    eval_coords = tuple( map(np.array, [lat, lon, r] ) )
    current_params = tuple( map(np.array, [lat_I, lon_I, r_I, Ie, In, Ir] ) )

    # turn intput into flattened arrays
    eval_coords_shape    = np.broadcast(*eval_coords).shape
    current_params_shape = np.broadcast(*current_params).shape
    eval_coords    = [np.ravel(_.reshape(eval_coords_shape   )) for _ in eval_coords]
    current_params = [np.ravel(_.reshape(current_params_shape)) for _ in current_params]

    N = eval_coords[0].size
    K = current_params[0].size

    # Construct 3 x N array, r_ecef, of N vectors pointing at evaluation points (ECEF)
    ph, th, rr = np.deg2rad(eval_coords[1]), np.deg2rad(90 - eval_coords[0]), eval_coords[2]
    r_ecef = rr * np.vstack((np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)))

    # Construct 3 x K array, s_ecef, of K vectors pointing at wedge intersection (ECEF)
    ph_I, th_I, r_I = np.deg2rad(current_params[1]), np.deg2rad(90 - current_params[0]), current_params[2]
    s_ecef = r_I * np.vstack((np.cos(ph_I) * np.sin(th_I), np.sin(ph_I) * np.sin(th_I), np.cos(th_I)))

    # Construct 3 x K array, j, of K unit vectors upward along the inclined leg of the current wedge (ECEF)
    j_enu  = np.vstack((current_params[3], current_params[4], current_params[5]))
    j_enu  = j_enu / np.linalg.norm(j_enu, axis = 0) # normalize
    signs = np.sign(j_enu[2])
    j_enu *= signs.reshape((1, -1)) # turn vectors upward if they point down
    j_ecef = enu_to_ecef(j_enu.T, current_params[1], current_params[0]).T

    # Find the lengths along j (from s) that are closest to evaluation points (N x K)
    t = np.einsum('in, ik -> nk', r_ecef, j_ecef) - np.sum(s_ecef * j_ecef, axis = 0).reshape((1, K))

    # Find vectors pointing at evaluation points from the closest point along j (3 x N x K):
    px = r_ecef[0].reshape((N, 1)) - s_ecef[0].reshape((1, K)) - j_ecef[0] * t
    py = r_ecef[1].reshape((N, 1)) - s_ecef[1].reshape((1, K)) - j_ecef[1] * t
    pz = r_ecef[2].reshape((N, 1)) - s_ecef[2].reshape((1, K)) - j_ecef[2] * t
    p_ecef = np.stack((px, py, pz))

    # Find distances between evaluation points and closest poinst along j (N X K):
    d = np.sqrt(px**2 + py**2 + pz**2)

    # normalized versions of p_ecef vectors:
    pp_ecef = p_ecef / d.reshape((1, N, K)) 

    # the magnetic field direction of inclined current line is pp_ecef cross j_ecef (3 x N x K array):
    eBx = pp_ecef[1] * j_ecef[2] - pp_ecef[2] * j_ecef[1] # cross product x component
    eBy = pp_ecef[2] * j_ecef[0] - pp_ecef[0] * j_ecef[2] # y component
    eBz = pp_ecef[0] * j_ecef[1] - pp_ecef[1] * j_ecef[0] # z component
    eB  = np.stack((eBx, eBy, eBz))            # 3 x N x K array with stacked components
    assert np.allclose(np.linalg.norm(eB, axis = 0), 1)

    # angle between the vectors p and and the vectors pointing from evaluation points to base (N x K)
    theta_1 = np.arctan(-t / d) # theta_1 in Griffiths fig 5.19 (theta_2 = pi/2 in our case)

    # magnetic field scaling factor (N x K):
    B_scale = MU0 / (4 * np.pi * d) * (1 - np.sin(theta_1))
    B_scale = B_scale.reshape((1, N, K))

    # (3 x N x K) array that map current magnitudes to ECEF components of the magnetic field:
    G_ecef = B_scale * eB

    # convert GB_ecef to enu - three matrices that are N x K:
    ph, th = ph.reshape((N, 1)), th.reshape((N, 1))
    G_e = -np.sin(ph)              * G_ecef[0] + np.cos(ph)              * G_ecef[1]
    G_n = -np.cos(th) * np.cos(ph) * G_ecef[0] - np.cos(th) * np.sin(ph) * G_ecef[1] + np.sin(th) * G_ecef[2]
    G_r =  np.sin(th) * np.cos(ph) * G_ecef[0] + np.sin(th) * np.sin(ph) * G_ecef[1] + np.cos(th) * G_ecef[2]

    if calculate_radial_leg:
        G_e_r, G_n_r, G_r_r = get_wedge_G_matrix(lat, lon, r, lat_I, lon_I, r_I, Ie*0, In*0, Ir, calculate_radial_leg = False)
        G_e, G_n, G_r = G_e - G_e_r, G_n - G_n_r, G_r - G_r_r

    return G_e, G_n, G_r


def get_CF_SECS_B_G_matrices_for_inclined_field(lat, lon, r, lat_secs, lon_secs, Be, Bn, Br, RI = RE + 110 * 1e3):
    """ Calculate matrix G that maps between CF SECS amplitudes and magnetic field, for inclined
        magnetic field lines. The magnetic field lines are modeled as semi-infinite lines, but with
        this function they are *not* assumed to be radial


    This function combines the standard SECS magnetic field, from get_SECS_B_G_matrices with the magnetic field
    of a current wedge that consists of two semi-inifinite field lines, calculated with get_wedge_G_matrix.

    Note
    ----
    This function does not (yet) support the singularity fix used in get_SECS_B_G_matrices, so be careful when
    evaluating the magnetic field close to either the radial or the inclined current lines

    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    r: array-like
        Array of radii of evaluation points. Flattened
        array must either have size 1, in which case one
        radius is used for all points, or have same size as 
        lat. Unit should be the same as RI 
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
    Be: array-like
        Array of main magnetic field eastward component at SECS pole
        locations. Same size as lat_secs
    Bn: array-like
        Array of main magnetic field northward component at SECS pole
        locations. Same size as lat_secs
    Br: array-like
        Array of main magnetic field radial component at SECS pole
        locations. Same size as lat_secs
    RI: float (optional)
        Radius of SECS poles. Default is Earth radius + 110,000 m

    Returns
    -------
    Ge: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the eastward magnetic field at (lat, lon) via 'Be = Ge.dot(m)'
    Gn: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the northward magnetic field at (lat, lon) via 'Bn = Gn.dot(m)'
    Gr: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the radial magnetic field at (lat, lon) via 'Br = Gr.dot(m)'
    """

    Ge_wedge, Gn_wedge, Gr_wedge = get_wedge_G_matrix(lat, lon, r, lat_secs, lon_secs, np.full_like(lat_secs, RI), Be, Bn, Br, calculate_radial_leg = True)
    Ge, Gn, Gr = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, RI = RI, current_type = 'curl_free')


    return Ge + Ge_wedge, Gn + Gn_wedge, Gr + Gr_wedge

