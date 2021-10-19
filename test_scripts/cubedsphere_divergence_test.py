""" Test numerical calculation of divergence in cubed sphere coords
"""
import numpy as np
import matplotlib.pyplot as plt
from secsy import CSgrid, CSprojection

d2r = np.pi / 180

lat0, lon0 = 60., 150. # center of projection
orientation = 45. # orientation of projection in east/north

N, M = 25, 25 # SH degree and order of the spherical harmonic used for testing

projection = CSprojection((lon0, lat0), orientation)
grid = CSgrid(projection, 3000, 2000, 5., 5.)
shape = grid.lat.shape
ph = grid.lon * np.pi / 180
th = (90 - grid.lat) * np.pi / 180

# Calculate SH derivatives (components) and divergence
P = np.cos(N * 2 * th) - 1
dP  = -N * 2 * np.sin(N * 2 * th)
dP2 = -N**2 * 4 * np.cos(N * 2 * th)
Y     = P  * (np.cos(M * ph) + np.sin(M * ph))
dYdth = dP * (np.cos(M * ph) + np.sin(M * ph))
dYdph = P * M * (np.cos(M * ph) - np.sin(M * ph))

div_analytic = -(np.cos(M * ph) + np.sin(M * ph)) / (np.sin(th) * grid.R**2) * \
                   ( (2 * N * np.sin(2 * N * th)*np.cos(th) + 4 * N**2 * np.sin(th) * np.cos(2 * N * th)) +
                      M**2 * (np.cos(2 * N * th) - 1) / np.sin(th)
                   )


je =  dYdph / np.sin(th) / grid.R
jn = -dYdth              / grid.R
j = np.vstack((je.flatten().reshape((-1, 1)), jn.flatten().reshape((-1, 1))))


# calcualte divergence numerically:
D = grid.divergence(S = 1, return_sparse = True)
div_num = D.dot(j).reshape(shape)

# plot comparison
fig, axs = plt.subplots(ncols = 3, figsize = (10, 5))
axs[0].contourf(grid.xi, grid.eta, div_analytic)
axs[0].set_title('Divergence analytic')
axs[1].contourf(grid.xi, grid.eta, div_num)
axs[1].set_title('Divergence numerical')
axs[2].scatter(div_analytic.flatten(), div_num.flatten())
axs[2].set_xlabel('analytic')
axs[2].set_ylabel('numerical')
   
plt.ion()
plt.show()
plt.pause(0.001)


