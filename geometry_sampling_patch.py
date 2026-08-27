"""Area-weighted, face-interior sampling for DoMINO's geometry (STL) point downsampling.

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

An earlier version of this patch fixed only that half of the problem: it kept sampling
existing VERTICES, just weighted by their triangle's area (via physicsnemo's own
shuffle_array, its existing weighted-sampling primitive). That is enough for a normally
-tessellated mesh, but it is NOT enough in general, and is not enough for at least one
real project geometry (ref_material/CASE_80.zarr): vertex-based sampling can only ever
place a point at a location where a vertex already exists, and a real CAD-derived STL
can have very few vertices along a large flat/curved face even where a few giant
triangles cover most of that face's area. Checked directly against that file: its
cylindrical walls have exactly 3 distinct Z-coordinates across all 96,144 vertices (the
two rims, essentially) -- meaning no re-weighting of which vertex gets picked can ever
put a sample anywhere but those two rims, however the weights are computed.

This version instead samples a point *on the face* of each selected triangle (uniform
barycentric interpolation of that triangle's 3 vertices), with triangles selected with
probability proportional to their own area (a standard "sample points on a mesh"
technique). This is strictly more capable than reweighting which vertex gets picked --
it can place a sample anywhere on a selected triangle's surface, including a giant
triangle's interior where no vertex exists at all. Verified against
ref_material/CASE_80.zarr: the previous (vertex-reweighting) version still put 0% of its
samples at any Z strictly between the two rims (no vertices exist there to pick); this
version puts ~64% of its samples there, uniformly spread across the wall's height,
matching the wall's actual area distribution.

Triangles are selected WITH replacement (torch.multinomial), unlike physicsnemo's
shuffle_array (without replacement) -- appropriate here since each draw generates an
independently random point even when the same (typically large) triangle is picked more
than once, which is exactly the desired behavior when one triangle's area is a large
fraction of the whole mesh's.

Imported by train.py / test.py / compute_statistics.py by default
(`import geometry_sampling_patch  # noqa: F401`) -- unlike cuml_knn_patch.py, this isn't
a version-compatibility workaround with a non-code alternative (there's no config option
to achieve area-weighted geometry sampling any other way in physicsnemo 1.3.0), and the
effect is large enough (85%+ vs ~15% area coverage at a typical sample budget, and the
only way at all to represent large-but-coarsely-tessellated faces) that it's treated as
a correctness fix rather than an opt-in. If you'd rather keep physicsnemo's original
uniform-per-vertex behavior, delete the three imports.

This changes what data the geometry encoder actually sees during training/eval --
comparing checkpoints trained with vs. without this patch is not apples-to-apples.
Sampled points are no longer necessarily original mesh vertices (they lie exactly ON the
mesh surface, just not always at a vertex) -- this is expected and is the whole point;
nothing elsewhere in this project's pipeline assumes geometry_coordinates are vertices.

Only applies when model.geometry_encoding_type includes "stl" (the geometry_coordinates
this affects feeds the STL-based geometry encoder) -- inert otherwise, and inert if
data.sampling is false (downsample_geometry itself is a no-op in that case, patched or
not).

Assumes stl_vertices is laid out as contiguous (v0, v1, v2) triples per triangle -- true
for every .zarr this project has been tested against. If your data doesn't follow that
convention (check `stl_coordinates.shape[0] == 3 * stl_centers.shape[0]` and that
`stl_faces` is a plain arange in your own .zarr, e.g. via
notebooks/data_distribution_analysis.ipynb-style direct zarr reads), this patch will
compute nonsense triangles without raising an error -- verify before relying on it.
"""

import torch

from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe


def _downsample_geometry_area_weighted(self, stl_vertices: torch.Tensor) -> torch.Tensor:
    """Drop-in, area-weighted, face-interior replacement for
    DoMINODataPipe.downsample_geometry. See module docstring for why sampling on
    triangle faces (not just at existing vertices) matters."""
    if not self.config.sampling:
        return stl_vertices

    n_samples = self.config.geom_points_sample

    v0 = stl_vertices[0::3]
    v1 = stl_vertices[1::3]
    v2 = stl_vertices[2::3]
    triangle_areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    tri_idx = torch.multinomial(triangle_areas, n_samples, replacement=True)

    r1 = torch.rand(n_samples, device=stl_vertices.device, dtype=stl_vertices.dtype)
    r2 = torch.rand(n_samples, device=stl_vertices.device, dtype=stl_vertices.dtype)
    reflect = (r1 + r2) > 1  # reflect points outside the unit triangle back inside it
    r1 = torch.where(reflect, 1 - r1, r1)
    r2 = torch.where(reflect, 1 - r2, r2)

    a, b, c = v0[tri_idx], v1[tri_idx], v2[tri_idx]
    return a + r1.unsqueeze(-1) * (b - a) + r2.unsqueeze(-1) * (c - a)


DoMINODataPipe.downsample_geometry = _downsample_geometry_area_weighted
