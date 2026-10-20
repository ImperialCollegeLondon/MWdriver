"""Masked wavefront meshes for interactive surface viewing."""

import numpy as np


SURFACE_RESULTS = {
    "surface_tilt_removed": ("Interactive 3D: Tilt removed", "tilt_removed"),
    "surface_defocus_removed": ("Interactive 3D: Tilt and defocus removed", "wavefront_aberration_nm"),
    "surface_residual": ("Interactive 3D: Residual wavefront", "residual_nm"),
}


def wavefront_surface_limits(tilt_removed_nm):
    values = np.ma.masked_invalid(np.ma.asarray(tilt_removed_nm, dtype=float)).compressed()
    if values.size == 0:
        raise ValueError("Tilt-removed wavefront contains no valid data.")
    lower, upper = np.percentile(values, [0.1, 99.1])
    if lower == upper:
        lower -= 1e-9
        upper += 1e-9
    return float(lower), float(upper)


def build_wavefront_surface(results, selected_key):
    wavelength = float(results.get("wavelength_nm", 632.8))
    factors = (wavelength / (2.0 * np.pi), 1.0, 2.0)
    surfaces = {}
    for (key, (_, source)), factor in zip(SURFACE_RESULTS.items(), factors):
        if source in results:
            surfaces[key] = np.ma.masked_invalid(
                np.ma.asarray(results[source], dtype=float)
            ) * factor
    surface = surfaces[selected_key]
    if surface.ndim != 2:
        raise ValueError("Wavefront surface must be a 2D array.")
    valid = ~np.ma.getmaskarray(surface)
    valid_y, valid_x = np.nonzero(valid)
    if valid_x.size < 3:
        raise ValueError("Not enough valid pixels for a wavefront surface.")
    rows = np.arange(valid_y.min(), valid_y.max() + 1)
    columns = np.arange(valid_x.min(), valid_x.max() + 1)
    grid_x, grid_y = np.meshgrid(columns, rows)
    indices = np.arange(grid_x.size).reshape(grid_x.shape)
    top_left = indices[:-1, :-1].ravel()
    top_right = indices[:-1, 1:].ravel()
    bottom_left = indices[1:, :-1].ravel()
    bottom_right = indices[1:, 1:].ravel()
    triangles = np.concatenate((
        np.column_stack((top_left, top_right, bottom_right)),
        np.column_stack((top_left, bottom_right, bottom_left)),
    ))
    invalid_sum = np.pad((~valid).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    invalid_cells = (
        invalid_sum[np.ix_(rows[1:] + 1, columns[1:] + 1)]
        - invalid_sum[np.ix_(rows[:-1], columns[1:] + 1)]
        - invalid_sum[np.ix_(rows[1:] + 1, columns[:-1])]
        + invalid_sum[np.ix_(rows[:-1], columns[:-1])]
    )
    triangles = triangles[~np.tile(invalid_cells.ravel() > 0, 2)]
    if not triangles.size:
        raise ValueError("No connected valid pixels for a wavefront surface.")
    used, faces = np.unique(triangles, return_inverse=True)
    heights = surface.filled(np.nan)[np.ix_(rows, columns)].ravel()[used]
    vertices = np.column_stack((heights, grid_y.ravel()[used], grid_x.ravel()[used]))
    limits = wavefront_surface_limits(surfaces.get("surface_tilt_removed", surface))
    extent = max(float(np.ptp(valid_x)), float(np.ptp(valid_y)), 1.0)
    return {
        "data": (vertices, faces.reshape(-1, 3), heights),
        "contrast_limits": limits,
        "scale": (extent / max(limits[1] - limits[0], 1e-9), 1.0, 1.0),
    }