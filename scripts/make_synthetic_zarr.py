"""Generate tiny synthetic .zarr cases matching ref_material/zarr_structure.md,
for local pipeline testing (see tests/run_smoke_test.sh) or as a concrete
example of the expected dataset layout. NOT real CFD/CHT data.

Each case is one <output_dir>/case_XXXX.zarr directory (a torch.utils.data.Dataset
item) containing:
    global_params_reference (6, 1) float32
    global_params_values    (6, 1) float32
    stl_areas          (n_tri,)      float32
    stl_centers        (n_tri, 3)    float32
    stl_coordinates    (3*n_tri, 3)  float32   -- 3 vertices per triangle, not deduplicated
    stl_faces          (3*n_tri,)    int32     -- indices into stl_coordinates (trivial, since
                                                   vertices above are already per-triangle)
    surface_areas          (n_pts,)    float32
    surface_fields          (n_pts, 1)  float32  -- temperature
    surface_mesh_centers    (n_pts, 3)  float32
    surface_normals         (n_pts, 3)  float32
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr

# Matches conf/config.yaml's variables.global_parameters order/reference values.
GLOBAL_PARAM_REFERENCE = np.array(
    [[300.0], [0.175], [300.0], [0.01], [300.0], [0.01]], dtype=np.float32
)


def _random_box_triangles(rng: np.random.Generator, n_tri: int):
    """Triangles with centers/normals/areas on the surface of a unit box."""
    centers = rng.uniform(-1.0, 1.0, size=(n_tri, 3)).astype(np.float32)
    axis = rng.integers(0, 3, size=n_tri)
    sign = rng.choice([-1.0, 1.0], size=n_tri).astype(np.float32)
    centers[np.arange(n_tri), axis] = sign  # snap one coordinate to the box face

    normals = np.zeros((n_tri, 3), dtype=np.float32)
    normals[np.arange(n_tri), axis] = sign

    tangent1 = np.zeros((n_tri, 3), dtype=np.float32)
    tangent1[np.arange(n_tri), (axis + 1) % 3] = 1.0
    tangent2 = np.cross(normals, tangent1)

    edge_len = rng.uniform(0.02, 0.08, size=(n_tri, 1)).astype(np.float32)
    coordinates = np.stack(
        [
            centers,
            centers + tangent1 * edge_len,
            centers + tangent2 * edge_len,
        ],
        axis=1,
    ).reshape(-1, 3)  # (3*n_tri, 3)

    areas = (0.5 * edge_len[:, 0] ** 2).astype(np.float32)
    return coordinates, centers, normals, areas


def _synthetic_temperature(points: np.ndarray, global_params: np.ndarray, rng: np.random.Generator):
    """A smooth, learnable synthetic temperature field: driven by distance
    from an inlet-like point plus the case's inlet temperature, so the
    pipeline has an actual (if fake) signal to fit during a smoke test."""
    inlet_temp = float(global_params[0, 0])
    dist = np.linalg.norm(points - np.array([-1.0, 0.0, 0.0]), axis=-1)
    noise = rng.normal(scale=0.5, size=points.shape[0]).astype(np.float32)
    temperature = inlet_temp - 20.0 * dist + noise
    return temperature.astype(np.float32).reshape(-1, 1)


def make_case(path: Path, n_tri: int, n_surface_pts: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    stl_coordinates, stl_centers, _, stl_areas = _random_box_triangles(rng, n_tri)
    stl_faces = np.arange(stl_coordinates.shape[0], dtype=np.int32)

    surface_coords, surface_centers, surface_normals, surface_areas = _random_box_triangles(
        rng, n_surface_pts
    )
    surface_mesh_centers = surface_centers  # one "point" per synthetic triangle center

    global_params_values = (
        GLOBAL_PARAM_REFERENCE * rng.uniform(0.8, 1.2, size=GLOBAL_PARAM_REFERENCE.shape)
    ).astype(np.float32)

    surface_fields = _synthetic_temperature(surface_mesh_centers, global_params_values, rng)

    if path.exists():
        shutil.rmtree(path)
    group = zarr.open_group(str(path), mode="w")
    arrays = {
        "global_params_reference": GLOBAL_PARAM_REFERENCE,
        "global_params_values": global_params_values,
        "stl_areas": stl_areas,
        "stl_centers": stl_centers,
        "stl_coordinates": stl_coordinates,
        "stl_faces": stl_faces,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,
        "surface_mesh_centers": surface_mesh_centers,
        "surface_normals": surface_normals,
    }
    for name, array in arrays.items():
        group.create_array(name, shape=array.shape, dtype=array.dtype)
        group[name][:] = array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-train", type=int, default=3)
    parser.add_argument("--n-val", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=1)
    parser.add_argument("--n-tri", type=int, default=200, help="STL triangle count per case")
    parser.add_argument(
        "--n-surface-pts", type=int, default=800, help="surface solution point count per case"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    case_idx = 0
    for split, n_cases in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_cases):
            case_path = split_dir / f"case_{i:04d}.zarr"
            make_case(case_path, args.n_tri, args.n_surface_pts, seed=args.seed + case_idx)
            print(f"wrote {case_path}")
            case_idx += 1


if __name__ == "__main__":
    main()
