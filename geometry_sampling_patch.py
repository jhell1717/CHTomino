"""Area-weighted sampling for DoMINO's geometry (STL) point downsampling.

physicsnemo's DoMINODataPipe.downsample_geometry
(physicsnemo/datapipes/cae/domino_datapipe.py) samples model.geom_points_sample points
uniformly at random from stl_coordinates -- the raw triangle VERTEX array (3 rows per
triangle, not deduplicated; confirmed for this project's .zarr files via
stl_coordinates.shape[0] == 3 * stl_centers.shape[0], with stl_faces a trivial arange
over that layout). Uniform-per-vertex sampling means every triangle contributes exactly
3 candidate points regardless of its area -- so a mesh with many small triangles
concentrated at edges/fillets/curved features and few large triangles on flat faces (a
completely normal result of most CAD meshing) gets systematically oversampled at those
edges and undersampled on the flat faces, however much of the object's actual surface
area those faces cover.

Confirmed against a real project .zarr (ref_material/CASE_80.zarr, 32048 STL triangles):
the largest 10% of triangles by count cover 93% of the mesh's total surface area, but
under current (uniform) sampling, a 5000-point sample only falls within triangles
covering ~15% of that area; area-weighting the same 5000-point budget instead covers
~87%. Visually this shows up exactly as reported: sampled geometry points cluster around
edges/vertices where the mesh happens to be finely triangulated, while large flat
surfaces are barely represented at all.

Unlike model.surface_sampling_algorithm (already configurable between "random" and
"area_weighted" for the SURFACE SOLUTION points, in physicsnemo's process_surface),
physicsnemo has no equivalent config option for the GEOMETRY points --
downsample_geometry unconditionally calls shuffle_array with no weights, and there's no
config flag to change that. This patch computes each triangle's area directly from
stl_vertices (via the standard cross-product formula -- no separate stl_areas array
needed) and passes it as per-vertex sampling weights to physicsnemo's own shuffle_array,
the same weighted-sampling primitive process_surface already uses for area-weighted
surface-point sampling.

Imported by train.py / test.py / compute_statistics.py by default
(`import geometry_sampling_patch  # noqa: F401`) -- unlike cuml_knn_patch.py, this isn't
a version-compatibility workaround with a non-code alternative (there's no config option
to achieve area-weighted geometry sampling any other way in physicsnemo 1.3.0), and the
effect is large enough (85%+ vs ~15% area coverage at a typical sample budget) that it's
treated as a correctness fix rather than an opt-in. If you'd rather keep physicsnemo's
original uniform-per-vertex behavior, delete the three imports.

This changes what data the geometry encoder actually sees during training/eval --
comparing checkpoints trained with vs. without this patch is not apples-to-apples.

Only applies when model.geometry_encoding_type includes "stl" (the geometry_coordinates
this affects feeds the STL-based geometry encoder) -- inert otherwise, and inert if
model.data.sampling is false (downsample_geometry itself is a no-op in that case, patched
or not).

Assumes stl_vertices is laid out as contiguous (v0, v1, v2) triples per triangle -- true
for every .zarr this project has been tested against. If your data doesn't follow that
convention (check `stl_coordinates.shape[0] == 3 * stl_centers.shape[0]` and that
`stl_faces` is a plain arange in your own .zarr, e.g. via
notebooks/data_distribution_analysis.ipynb-style direct zarr reads), this patch will
compute nonsense weights without raising an error -- verify before relying on it.
"""

import torch

from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe, shuffle_array


def _downsample_geometry_area_weighted(self, stl_vertices: torch.Tensor) -> torch.Tensor:
    """Drop-in, area-weighted replacement for DoMINODataPipe.downsample_geometry."""
    if not self.config.sampling:
        return stl_vertices

    geometry_points = self.config.geom_points_sample

    v0 = stl_vertices[0::3]
    v1 = stl_vertices[1::3]
    v2 = stl_vertices[2::3]
    triangle_areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    vertex_weights = triangle_areas.repeat_interleave(3)

    geometry_coordinates_sampled, _ = shuffle_array(
        stl_vertices, geometry_points, weights=vertex_weights
    )
    if geometry_coordinates_sampled.shape[0] < geometry_points:
        raise ValueError("Surface mesh has fewer points than requested sample size")

    return geometry_coordinates_sampled


DoMINODataPipe.downsample_geometry = _downsample_geometry_area_weighted
