from importlib import reload
from secsy import cubedsphere
reload(cubedsphere)
from apexpy import Apex
import numpy as np
import datetime as dt

lon_centre= 17.5
lat_centre= 68.1
A = Apex(date=dt.datetime(2008, 6, 1, 0, 0, 0))
f1, f2 = A.basevectors_qd(lat_centre, lon_centre, 0, coords = 'geo')
qd_north = f2 / np.linalg.norm(f2)
East, North= qd_north[0], qd_north[1]
Gridproj=  cubedsphere.CSprojection((lon_centre, lat_centre), [East, North])
node_grid= cubedsphere.CSgrid(Gridproj, 3700, 2200, 70., 70.)
node_f1, node_f2= A.basevectors_qd(node_grid.lat.flatten(), node_grid.lon.flatten(), 9, coords='geo')
node_grid.get_Le_Ln()