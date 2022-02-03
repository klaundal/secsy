Overview
========
secsy - a Python module for working with Spherical Elementary Current Systems (SECS) and cubed sphere projection. If and only if those words made sense to you, you may find this module useful. Here are some main features:

- ``get_SECS_J_G_matrices``: Function that calculate matrices that relate SECS amplitudes and current densities (curl-free or divergence-free), with optional correction for singularities at the poles
- ``get_SECS_B_G_matrices``: Function that calculates matrices that relate SECS amplitudes and magnetic fields
- ``CSprojection``: Class that sets up a cubed sphere projection, with a single cube phase centered on an arbitrary longitude and latitude, and with an arbitrary orientation
- ``CSgrid``: Class that sets up a grid with a given projection, resolution, and extent. The class contains functions that make differentiation matrices that take into account distortion effects. 

Check doc strings for more information, or look at the example notebook. Or get in touch.

The SECS functions and cubed sphere stuff can be used independently. They are in the same module because we find that cubed sphere grids work well for SECS analysis. 

Dependencies
============
You should have the following modules installed:

- numpy
- scipy
- cartopy

Install
=======
Use git. Clone the repository like this::

    git clone https://github.com/klaundal/secsy

If the secsy folder is somewhere Python knows about (for example your working directory, or in the PYTHONPATH environment variable), it will work as demonstrated above.


References
==========
Here is the reference for Cubed Sphere projections:
C. Ronchi, R. Iacono, P.S. Paolucci, The “Cubed Sphere”: A New Method for the Solution of Partial Differential Equations in Spherical Geometry, Journal of Computational Physics, Volume 124, Issue 1, 1996, Pages 93-114, ISSN 0021-9991, https://doi.org/10.1006/jcph.1996.0047.

Here is a good reference for SECS analysis: 
Vanhamäki H., Juusola L. (2020) Introduction to Spherical Elementary Current Systems. In: Dunlop M., Lühr H. (eds) Ionospheric Multi-Spacecraft Analysis Tools. ISSI Scientific Report Series, vol 17. Springer, Cham. https://doi.org/10.1007/978-3-030-26732-2_2
