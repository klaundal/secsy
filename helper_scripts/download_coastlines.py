""" 
Script to download coastline coordinates with cartopy, 
and save as numpy arrays.

"""

import cartopy
import numpy as np


def get_projected_coastlines(**kwargs):
    """ generate coastlines in projected coordinates """
    
    try:
        import cartopy.io.shapereader as shpreader
    except ModuleNotFoundError:
        ModuleNotFoundError('Package missing. cartopy is required for downloading coastlines')


    if 'resolution' not in kwargs.keys():
        kwargs['resolution'] = '50m'
    if 'category' not in kwargs.keys():
        kwargs['category'] = 'physical'
    if 'name' not in kwargs.keys():
        kwargs['name'] = 'coastline'

    shpfilename = shpreader.natural_earth(**kwargs)
    reader = shpreader.Reader(shpfilename)
    coastlines = reader.records()
    multilinestrings = []
    for coastline in coastlines:
        if coastline.geometry.geom_type == 'MultiLineString':
            multilinestrings.append(coastline.geometry)
            continue
        lon, lat = np.array(coastline.geometry.coords[:]).T 

        yield lat, lon
            

    for mls in multilinestrings:
        for ls in mls:
            lon, lat = np.array(ls.coords[:]).T

            yield lat, lon


for res in ['50m', '110m']:
    coords = [np.vstack(c) for c in get_projected_coastlines(resolution = res)]
    np.savez('../data/coastlines_' + res + '.npz', *coords)