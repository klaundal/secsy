#!/usr/bin/env python3
"""
Download and pack Natural Earth map features for secsy.

By default this writes the bundled feature archives used by
``secsy.natural_earth``:

    src/secsy/data/natural_earth_npz/
        natural_earth_10m_all.npz
        natural_earth_50m_all.npz
        natural_earth_110m_all.npz

The script uses Cartopy's Natural Earth downloader/cache and therefore requires
``cartopy`` plus its geometry dependencies.
"""

import argparse
import json
from pathlib import Path

import numpy as np


RESOLUTIONS = ("10m", "50m", "110m")

LAYERS = {
    "land": ("physical", "land"),
    "ocean": ("physical", "ocean"),
    "coastline": ("physical", "coastline"),
    "lakes": ("physical", "lakes"),
    "rivers_lake_centerlines": ("physical", "rivers_lake_centerlines"),
    "borders": ("cultural", "admin_0_boundary_lines_land"),
    "states_provinces": ("cultural", "admin_1_states_provinces_lines"),
}


def default_output_dir():
    """
    Return the default output directory for bundled Natural Earth archives.

    The path is resolved relative to this script, so it works when the script is
    executed from any current working directory.
    """
    return Path(__file__).resolve().parents[1] / "data" / "natural_earth_npz"


def iter_parts(geom):
    """
    Yield flat coordinate arrays from a Cartopy/Shapely geometry.

    Parameters
    ----------
    geom : shapely geometry
        Geometry returned by Cartopy's Natural Earth shapereader.

    Yields
    ------
    part_type, coords : tuple
        ``part_type`` describes the geometry part, for example ``'line'`` or
        ``'polygon_exterior'``. ``coords`` is an ``(N, 2)`` longitude/latitude
        array.
    """
    if geom is None:
        return

    gtype = geom.geom_type

    if gtype == "Point":
        yield "point", np.asarray([geom.coords[0]], dtype=np.float64)
    elif gtype == "MultiPoint":
        for part in geom.geoms:
            yield from iter_parts(part)
    elif gtype == "LineString":
        yield "line", np.asarray(geom.coords, dtype=np.float64)
    elif gtype == "MultiLineString":
        for part in geom.geoms:
            yield from iter_parts(part)
    elif gtype == "Polygon":
        yield "polygon_exterior", np.asarray(geom.exterior.coords, dtype=np.float64)
        for ring in geom.interiors:
            yield "polygon_hole", np.asarray(ring.coords, dtype=np.float64)
    elif gtype == "MultiPolygon":
        for part in geom.geoms:
            yield from iter_parts(part)
    elif gtype == "GeometryCollection":
        for part in geom.geoms:
            yield from iter_parts(part)


def pack_segments(parts):
    """
    Pack geometry parts into flat arrays for NPZ storage.

    Parameters
    ----------
    parts : iterable
        Iterable of ``(part_type, coords)`` tuples returned by ``iter_parts``.

    Returns
    -------
    coords, offsets, part_types : tuple of arrays
        ``coords`` contains all coordinates for a layer. Segment ``i`` is
        recovered as ``coords[offsets[i]:offsets[i + 1]]`` and has type
        ``part_types[i]``.
    """
    coords = []
    offsets = [0]
    part_types = []

    for part_type, arr in parts:
        if arr.size == 0:
            continue
        coords.append(arr)
        offsets.append(offsets[-1] + len(arr))
        part_types.append(part_type)

    if coords:
        coords = np.vstack(coords)
    else:
        coords = np.empty((0, 2), dtype=np.float64)

    return (
        coords,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(part_types, dtype="U32"),
    )


def export_resolution(resolution, outdir, layers):
    """
    Export selected Natural Earth layers for one resolution.

    Parameters
    ----------
    resolution : {'10m', '50m', '110m'}
        Natural Earth resolution to download and pack.
    outdir : pathlib.Path
        Directory where the compressed NPZ archive is written.
    layers : iterable of str
        Layer names from ``LAYERS`` to include.

    Returns
    -------
    pathlib.Path
        Path of the written NPZ archive.
    """
    import cartopy.io.shapereader as shpreader

    outpath = outdir / f"natural_earth_{resolution}_all.npz"

    npz = {}
    metadata = {
        "resolution": resolution,
        "format": {
            "<layer>_coords": "Flat [N, 2] lon/lat coordinate array.",
            "<layer>_offsets": (
                "Offsets into coords. Segment i is "
                "coords[offsets[i]:offsets[i+1]]."
            ),
            "<layer>_part_types": "Geometry part type for each segment.",
            "<layer>_records_json": (
                "JSON list containing source attributes and segment ranges per "
                "source record."
            ),
        },
        "layers": {},
    }

    for layer in layers:
        category, name = LAYERS[layer]
        print(f"[{resolution}] {layer}")

        shp = shpreader.natural_earth(
            resolution=resolution,
            category=category,
            name=name,
        )
        reader = shpreader.Reader(shp)

        all_parts = []
        records_json = []

        for record_index, record in enumerate(reader.records()):
            geom = record.geometry
            if geom is None:
                continue

            start_part = len(all_parts)
            attrs = dict(record.attributes)

            for part_type, arr in iter_parts(geom):
                all_parts.append((part_type, arr))

            end_part = len(all_parts)
            records_json.append(
                {
                    "record_index": record_index,
                    "attributes": attrs,
                    "part_start": start_part,
                    "part_end": end_part,
                }
            )

        coords, offsets, part_types = pack_segments(all_parts)

        npz[f"{layer}_coords"] = coords
        npz[f"{layer}_offsets"] = offsets
        npz[f"{layer}_part_types"] = part_types
        npz[f"{layer}_records_json"] = np.asarray(
            json.dumps(records_json, ensure_ascii=False)
        )

        metadata["layers"][layer] = {
            "category": category,
            "name": name,
            "source_shapefile": str(shp),
            "n_points": int(coords.shape[0]),
            "n_parts": int(len(offsets) - 1),
            "n_records": int(len(records_json)),
        }

    npz["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))

    np.savez_compressed(outpath, **npz)
    print(f"saved {outpath}\n")
    return outpath


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and pack Natural Earth features for secsy."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output directory. Defaults to secsy's bundled data folder.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=RESOLUTIONS,
        default=list(RESOLUTIONS),
        help="Natural Earth resolutions to export.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=tuple(LAYERS),
        default=list(LAYERS),
        help="Feature layers to export.",
    )
    return parser.parse_args()


def main():
    """Download and pack the requested Natural Earth feature archives."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for resolution in args.resolutions:
        export_resolution(resolution, args.output_dir, args.layers)


if __name__ == "__main__":
    main()
