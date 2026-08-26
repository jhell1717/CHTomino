# Surface-Temperature DoMINO Surrogate

A [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) DoMINO surrogate model
that predicts steady-state surface temperature on a geometry from its shape
and a set of boundary-condition parameters. Refactored from an internal
DrivAer aerodynamics (pressure/velocity/shear) surrogate down to what this
dataset actually is: **surface-only, one scalar field (temperature), no
volume solution.**

Targets **PhysicsNeMo v1.3.0** (`nvidia-physicsnemo` on PyPI). Verified
end-to-end (compute_statistics -> train -> test) against the real v1.3.0
package on a CPU-only machine using a synthetic dataset -- see
"Validation performed" below.

## Dataset

Each training/validation/test case is one `*.zarr` directory. Layout (see
`ref_material/zarr_structure.md`):

| key | shape | meaning |
|---|---|---|
| `global_params_reference` | (6, 1) | non-dimensionalization reference values, one row per boundary condition |
| `global_params_values` | (6, 1) | this case's boundary condition values (same order) |
| `stl_areas` | (n_tri,) | STL triangle areas (geometry mesh, for the geometry encoder) |
| `stl_centers` | (n_tri, 3) | STL triangle centers |
| `stl_coordinates` | (3·n_tri, 3) | STL triangle vertices (3 per triangle) |
| `stl_faces` | (3·n_tri,) | flat vertex indices into `stl_coordinates` |
| `surface_areas` | (n_pts,) | solution mesh element areas |
| `surface_fields` | (n_pts, 1) | **temperature** (the training target) |
| `surface_mesh_centers` | (n_pts, 3) | solution mesh point coordinates |
| `surface_normals` | (n_pts, 3) | solution mesh point normals |

The 6 `global_params_*` rows correspond, in order, to
`conf/config.yaml`'s `variables.global_parameters` block (`Temp_inlet`,
`Mdot_inlet`, `Temp_inlet_A`, `Mdot_inlet_A`, `Temp_inlet_B`, `Mdot_inlet_B`).
If your data uses different boundary conditions or a different count, edit
that block to match -- the model's global-feature input dimension is derived
from it (`utils.get_num_vars`).

Point each of `data.input_dir`, `data.input_dir_val`, `eval.test_path` at a
directory containing one subdirectory of `*.zarr` cases per split.

**Before training, also set `data.bounding_box` / `data.bounding_box_surface`**
in `conf/config.yaml` to your geometry's real physical coordinate range (see
the `TODO` comment there) -- the placeholders are copied from the reference
DrivAer config and are almost certainly wrong for your geometry.

## Setup

```bash
python3.12 -m venv .venv   # 3.10-3.12; torch/physicsnemo don't yet support 3.14
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or a cuXXX index for a GPU box
pip install -r requirements.txt
```

### Environment notes

- **`nvidia-dali` is required to *import* `physicsnemo.datapipes.cae`, even
  though DoMINO's `.zarr` path never uses it.** `physicsnemo.datapipes.cae`'s
  `__init__.py` unconditionally imports its (unrelated) DALI-based
  `MeshDatapipe`, which raises `ImportError` at import time if
  `nvidia-dali` isn't installed. On a real NVIDIA GPU / Linux training box,
  just `pip install nvidia-dali-cudaXXX` (matching your CUDA version) and
  this is a non-issue. On a machine without an NVIDIA GPU (e.g. this was
  developed and validated on a Mac with no GPU), `nvidia-dali` has no
  installable wheel, so `devtools/dali_stub/` provides a minimal stand-in
  package -- just enough attribute surface for the unrelated `MeshDatapipe`
  class body to import successfully; nothing in it is ever called by this
  project. Put it on `PYTHONPATH` when running locally:
  ```bash
  export PYTHONPATH="$(pwd)/devtools/dali_stub"
  ```
  Don't use this on the real training cluster -- install the real package
  there instead.

- **`warp-lang` must stay pinned to `1.12.0`** (see `requirements.txt`) --
  bisected directly against two conflicting upstream constraints:
  physicsnemo 1.3.0's neighbor/SDF code breaks against `warp-lang >= 1.13`
  (`AttributeError: module 'warp' has no attribute 'context'` at import
  time), while `warp-lang < 1.8.1` fails on a CUDA driver reporting CUDA
  13.0 (`Warp CUDA error: Failed to get driver entry point
  'cuDeviceGetUuid'`, a confirmed upstream bug,
  [NVIDIA/warp#940](https://github.com/NVIDIA/warp/issues/940), fixed in
  1.8.1). `1.12.0` is the newest release satisfying both. Neither of these
  is specific to this project.

- **No NVIDIA GPU here.** `data.gpu_preprocessing` / `data.gpu_output` /
  `train.amp.enabled` all default to GPU-oriented settings in
  `conf/config.yaml` for when you move this to a real training cluster.
  Running locally without a GPU works out of the box regardless --
  `torch.amp.autocast`/`GradScaler` auto-disable themselves when no CUDA
  device is present, and DoMINO's SDF/kNN preprocessing (via NVIDIA Warp)
  has a CPU fallback -- it's just much slower than on a GPU. Set
  `data.gpu_preprocessing=false data.gpu_output=false` explicitly if you
  want to be sure preprocessing stays off the GPU on a mixed setup.

## Usage

```bash
# 1. One-time: compute normalization statistics over the training set
python compute_statistics.py

# 2. Train
python train.py

# 3. Evaluate the best checkpoint against a held-out .zarr test set
python test.py
```

All three are [Hydra](https://hydra.cc) apps reading `conf/config.yaml`;
override anything on the command line, e.g.
`python train.py train.epochs=200 model.surface_points_sample=20000`.

`train.py` writes per-run outputs under `outputs/<project.name>/<exp_tag>/`:
TensorBoard logs, an MLflow run (tracked in `<project_dir>/mlflow.db`), the
fully-resolved config, and checkpoints under `models/` (periodic, every
`train.checkpoint_interval` epochs) and `models/best_model/` (whenever
validation loss improves -- this is what `test.py` loads by default).

`test.py` writes, per test case, a point-cloud `.vtp` with
`TemperaturePred`/`TemperatureTrue` fields and a predicted-vs-true parity
plot, under `eval.save_path`.

### If relative-L2 looks good but predicted-vs-true shows no diagonal trend

`model.normalization` (min-max or mean/std) is computed *globally* across all
training cases (`compute_statistics.py`). If temperature shifts a lot
case-to-case (different boundary conditions) relative to how much it actually
varies *within* one case's surface, that global range -- and therefore the
training loss and the plain relative-L2 metric -- is dominated by the
cross-case shift, not the spatial pattern. A model can then score a good
relative L2 while barely resolving the actual spatial pattern, sometimes with
predicted values scattered more widely than the true field and uncorrelated
with it (a "blob" instead of a diagonal in the parity plot). `test.py` prints
a second metric, `l2_surf_temperature_centered` (mean-centered per case,
isolating spatial-pattern accuracy from baseline accuracy) -- if it's much
worse than the plain number, this is what's happening.

Fix: reframe the target as a delta from a boundary-condition reference
temperature instead of absolute temperature, so neither the loss nor the
global normalization range is dominated by the cross-case shift:

```bash
python scripts/rebase_surface_temperature.py data/train data/train_delta
python scripts/rebase_surface_temperature.py data/val data/val_delta
python scripts/rebase_surface_temperature.py data/test data/test_delta
```

Then point `data.input_dir`/`data.input_dir_val`/`eval.test_path` at the new
`*_delta` directories, set `data.temperature_reference_key` to whichever
boundary condition you subtracted (so `test.py` adds it back for
interpretable `.vtp`/plot output -- metrics stay computed on the delta
values), delete the now-stale `scaling_factors.pkl`, and retrain from
scratch. See that script's docstring for the full reasoning and why the
reference has to come from a boundary condition, not the target field
itself. A checkpoint trained on absolute temperature is not compatible with
delta-temperature data or vice versa.

## Moving to a real NVIDIA GPU (e.g. an H100 VM)

This project targets GPU training -- `conf/config.yaml`'s
`data.gpu_preprocessing`/`data.gpu_output`/`train.amp.enabled` all default to
`true` already, and the code path for a single GPU (`world_size == 1`, no
`DistributedDataParallel` wrapping) is structurally the same path exercised
by the CPU smoke test, just with `dm.device.type == "cuda"` instead of
`"cpu"`. An H100 (Hopper, CUDA 12.x) is a completely standard target for
DoMINO training -- nothing here is CPU-specific or needs changing for it.

What to do differently there vs. the local setup described above:

1. **Install a CUDA build of torch**, not the CPU one -- see the comment in
   `requirements.txt` (e.g. the `cu124` index for CUDA 12.4).
2. **Install the real `nvidia-dali-cuda120`** (or `-cuda110`, matching your
   CUDA major version) instead of using `devtools/dali_stub/` -- see
   `requirements.txt` and "Environment notes" above. Do **not** put
   `devtools/dali_stub` on `PYTHONPATH` there; it's a local/no-GPU-only
   shim, and if it's importable it'll silently shadow the real DALI package.
3. **Keep the `warp-lang==1.12.0` pin.** See "Environment notes" above --
   it's bisected to satisfy two conflicting upstream constraints at once
   (physicsnemo 1.3.0 breaks above 1.12.x; CUDA-13.0-reporting drivers, which
   is what you're likely to hit on a real GPU box, break below 1.8.1). If
   you still see `Warp CUDA error: Failed to get driver entry point
   'cuDeviceGetUuid'` with this pin in place, your driver may report a CUDA
   version outside what's been verified here -- check `nvidia-smi`'s
   reported CUDA version against [NVIDIA/warp#940](https://github.com/NVIDIA/warp/issues/940).
4. **Install RAPIDS cuML -- required, not optional, for any non-trivial mesh
   size.** `physicsnemo.utils.neighbors.knn()` (used by `DoMINODataPipe` on
   every case, for both training and evaluation) picks its backend based on
   what's importable: on CUDA points it uses cuML if available, otherwise it
   silently falls back to an *unchunked* brute-force torch implementation
   that materializes the full (N_points x N_queries x 3) pairwise-distance
   tensor in one allocation. At real surface-mesh sizes (hundreds of
   thousands of points) that's hundreds of GB to multiple TB -- it will OOM
   even on an H100, regardless of batch size or sample count, the moment
   `N_points x N_queries` gets large. There's no config flag for this; it's
   purely about whether `cuml` is importable in your environment.

   physicsnemo 1.3.0's cuML backend manually builds a `cuml.Handle(...)` to
   bind the search to PyTorch's current CUDA stream. RAPIDS removed `Handle`
   from `cuml`'s top-level namespace at some point after physicsnemo 1.3.0
   was built (confirmed: physicsnemo's own fix for this exact bug, in
   v2.1.1, drops `Handle` entirely and requires `cuml>=26.2.0`), so
   `pip install cuml-cu12` (unpinned -- physicsnemo's own floor is just
   `>=24.0.0`) grabbing today's latest release hits
   `AttributeError: module cuml has no attribute Handle` the first time
   `DoMINODataPipe` runs a kNN search. physicsnemo 1.3.0's own `Dockerfile`
   is built on `nvcr.io/nvidia/pytorch:25.09-py3` (released 2025-11-18), so
   it was actually tested against a **25.x** RAPIDS release, not an old 24.x
   one. Install a pinned 25.x point release instead of an unpinned one (see
   `requirements.txt`):
   ```bash
   pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.10.*"  # or cuml-cu11 for CUDA 11.x
   ```
   Verify it's actually being picked up before running anything expensive:
   ```bash
   python -c "from physicsnemo.utils.neighbors.knn._cuml_impl import CUML_AVAILABLE; print(CUML_AVAILABLE)"
   python -c "import cuml; print(cuml.Handle)"
   ```
   the first should print `True`, the second should print a class, not raise
   `AttributeError`. No code changes needed once this is true --
   `knn(backend="auto")` dispatches to it automatically.

   `25.10.*` is a well-supported guess, not a guarantee -- if `cuml.Handle`
   still doesn't exist, try adjacent point releases (`25.08.*`, `25.06.*`,
   ... run `pip index versions cuml-cu12` to see what's available) until one
   works.

   `cuml_knn_patch.py` in this repo is a code-level fallback for this
   specific bug, already imported at the top of `train.py`/`test.py`/
   `compute_statistics.py` -- it reimplements physicsnemo's kNN dispatch but
   routes the cuML backend through `cuml.neighbors.NearestNeighbors`
   directly instead of a manually built `Handle`, matching physicsnemo's own
   later (v2.1.1) fix. It's a no-op unless cuML is actually available and
   points are on CUDA, so it's harmless to leave in even if the version pin
   above resolves things on its own -- but it's unverified the same way as
   everything else here: there's no cuML/GPU in the environment this project
   was built in.

   **If you hit `AttributeError: type object 'BaseEstimator' has no
   attribute '_get_default_requests'`**: this is a different bug, once
   `import cuml` itself succeeds. `import cuml` transitively imports
   `cuml.accel` (an unrelated sklearn-acceleration feature, dragged in even
   though nothing here uses it), and cuml 25.10's
   `cuml/accel/estimator_proxy.py` directly calls
   `BaseEstimator._get_default_requests` (verified against cuml's own
   source). Checked against scikit-learn's source across releases: that
   method exists in scikit-learn 1.3.x-1.7.x and was **removed starting in
   1.8.0** (not 1.9 -- an earlier version of this note said 1.9, verify
   against source rather than trust that if it still doesn't work). cuml
   only declares `scikit-learn>=1.5` (no upper bound), so any unpinned `pip
   install` in the environment can silently pull scikit-learn up to 1.8+ --
   including ones unrelated to cuml itself. Fix:
   ```bash
   pip install "scikit-learn<1.8"
   ```
   (also pinned, commented, in `requirements.txt`). If this keeps
   recurring after later `pip install`s in the same environment, that's
   this same silent-upgrade pattern happening again, not a new issue.

   **If you hit `ModuleNotFoundError: No module named 'cupy'` despite
   `cupy-cuda12x` showing as installed**: `cupy-cuda12x` is the pip package
   name; the import is always just `cupy`. First confirm you're checking the
   same Python (`python -c "import sys; print(sys.executable)"` vs. `pip
   show cupy-cuda12x`'s `Location:`). If those match and `python -c "import
   cupy"` still fails on its own, the install itself is broken -- likely
   fallout from having both `cupy-cuda12x` and `cupy-cuda13x` installed at
   once (both claim the same `cupy` import namespace). Purge and reinstall:
   ```bash
   pip uninstall -y cupy-cuda12x cupy-cuda13x cupy
   pip cache purge
   pip install cupy-cuda12x
   ```

   **If you can't get cuML working at all** and need to unblock evaluation
   in the meantime: `test.py` defaults to evaluating each case's *full*
   surface mesh, which is exactly what makes the brute-force fallback OOM
   (`full_mesh_points x full_mesh_points`). Set `eval.sampling=true` and
   `eval.surface_points_sample=<N>` (see `conf/config.yaml` for the sizing
   formula) to evaluate a bounded random subset of each case's surface
   instead -- trades complete per-case coverage for something that actually
   fits the brute-force fallback's memory, e.g.:
   ```bash
   python test.py eval.sampling=true eval.surface_points_sample=4096
   ```
   This does not touch training (`train.py` always uses cuML if available,
   the brute-force fallback otherwise, at `model.surface_points_sample`, and
   OOMs the same way without cuML -- there's no equivalent workaround for
   training, since a representative training sample can't be shrunk this far
   without hurting the model). Switch back to `eval.sampling=false` once
   cuML is working, for real full-mesh evaluation.
5. Leave `data.gpu_preprocessing` / `data.gpu_output` / `train.amp.enabled`
   as `true` (the checked-in defaults) rather than the `false` overrides used
   for the local CPU smoke test.
6. **Set `data.bounding_box` / `data.bounding_box_surface`** to your real
   geometry's coordinate range (see the Dataset section above) and point
   `data.input_dir` / `data.input_dir_val` / `eval.test_path` at your real
   `.zarr` data.
7. Run `tests/run_smoke_test.sh` there first (with the real DALI and cuML
   installed, no stub) as a fast sanity check that the environment itself is
   sound, before a full training run.

**What's genuinely unverified**, since no GPU was available while building
this: the GPU-specific code paths themselves (device transfer,
`GradScaler`/`autocast` actually running in float16, `pynvml` memory
logging, the cuML-backed `knn()` path above), full-scale hyperparameters
(`interp_res: [128, 64, 64]`, `surface_points_sample: 70000`,
`geom_points_sample: 30000` -- standard DoMINO-scale settings, but never run
here even on the tiny synthetic data, so unconfirmed for memory fit /
throughput on your setup), NVIDIA DALI's own import behavior (untestable
without a real install), and everything about real CFD/CHT data (only
synthetic data was used). If something breaks on the VM, it's most likely
one of those, not the surface-only pipeline logic itself (data loading,
model construction/forward/backward, loss, checkpointing, evaluation), which
the smoke test does cover.

## What changed from the reference scripts (`ref_material/`)

The reference scripts were a DrivAer aerodynamics surrogate (pressure +
3-component wall shear stress, optionally a volume solution, optional
physics-informed Navier-Stokes residual losses, drag/lift/side-force integral
losses) adapted partway toward this CHT/temperature dataset, with the two
still mixed together. Since this dataset (`ref_material/zarr_structure.md`)
has **one scalar surface field and no volume solution at all**, this
refactor commits fully to that:

- **Removed**: the volume branch and `model_type: combined`/`volume` (the
  library's `DoMINO` model still *constructs* volume-geometry submodules
  internally regardless of `model_type` -- that's an upstream
  implementation detail, not something this project's config can avoid --
  but no volume data ever flows through this pipeline); the physics-informed
  Navier-Stokes residual loss (`compute_physics_loss`,
  `IncompressibleNavierStokes`) and `train.add_physics_loss`, which don't
  apply to a temperature field; the drag/lift/side-force integral losses
  (`integral_loss_fn`, `drag_loss_fn`, `lift_loss_fn`), which index into
  pressure + 3 shear components that don't exist here; the raw
  `*_solution.h5`/`*_geom.stl` ingestion path (`raw_data_utils.py`,
  `raw_utils.py`, and `ref_material/test.txt`'s approach) -- this project's
  data is already `.zarr`, read directly via PhysicsNeMo's built-in
  `CAEDataset`/`DoMINODataPipe`, which is both simpler and the officially
  supported path; the Optuna HPO config block (no driver script referenced
  it); a duplicated commented-out copy of the entire script body left in
  `ref_material/compute_statistics.txt`; the domain-parallel (tensor-sharded)
  training path -- multi-GPU still works via standard `DistributedDataParallel`,
  just not sharding a single case's grid/points across GPUs, which is an
  advanced feature not needed for a working prototype.
- **Fixed**: `train.py` saved periodic checkpoints but never actually wrote
  anything to `models/best_model/`, despite `eval.*` expecting to load a
  best-model checkpoint from there -- it now saves one whenever validation
  loss improves. `train.py` also referenced an undefined `vol_factors`
  variable inside the (dead) physics-loss branch. `torch.cuda.nvtx.range()`
  raises `RuntimeError` on a CPU-only PyTorch build -- profiling markers now
  degrade to no-ops off GPU (`utils.nvtx_range`) instead of crashing.
  `autocast`/checkpoint-loading device handling was hardcoded to `"cuda"` in
  a few places; now derived from `DistributedManager().device`, so it also
  runs correctly on CPU.
- **Kept**: MLflow + TensorBoard logging, AMP/grad-clipping, LR scheduling,
  the area-weighted auxiliary loss term, checkpoint resume, the historical
  (`eval.model_config`) evaluation path for comparing against an older run's
  architecture/normalization.

`test.py` is a new script (the provided `ref_material/test.txt` evaluated
directly against raw `*_solution.h5`/`*_geom.stl` files, which isn't this
project's data format -- see previous paragraph). `ref_material/pred_from_stl.py`
(bare-STL, no-ground-truth inference) wasn't provided and hasn't been
recreated; the closest equivalent today is `test.py` against a `.zarr` case
that has no `surface_fields` written for it (`get_keys_to_read`'s
`get_ground_truth=False` path already supports this) -- worth adding as a
dedicated script if you need it.

## Validation performed

No NVIDIA GPU is available in this environment, so the model has not been
trained on GPU or on real data. What **was** verified, running the actual
`nvidia-physicsnemo==1.3.0` package (not a mock) end-to-end on CPU:

```bash
tests/run_smoke_test.sh
```

This generates a tiny synthetic `.zarr` dataset matching the exact structure
above (`scripts/make_synthetic_zarr.py` -- not real CFD/CHT data, just
correctly-shaped arrays with a smooth synthetic temperature field), then runs
`compute_statistics.py` -> `train.py` (2 epochs, tiny model/grid/sampling
sizes) -> `test.py`, and checks that every step completes and produces the
expected checkpoint/`.vtp`/plot artifacts. It confirms: the `.zarr` data
loads correctly through PhysicsNeMo's `CAEDataset`/`DoMINODataPipe`; the
`DoMINO` model builds and runs a forward+backward pass in surface-only mode;
loss/L2-metric computation, checkpointing, checkpoint resume, and evaluation
all run without error and produce finite losses.

**Not verified**: prediction accuracy on real geometry/data (the synthetic
data has no physical meaning), multi-GPU behavior, and performance/memory
at the full point counts and grid resolution in the shipped `conf/config.yaml`
(the smoke test uses a deliberately tiny model to run in seconds on a CPU).

## Layout

```
conf/config.yaml         Hydra config (data paths, model hyperparameters, training)
compute_statistics.py    One-time: compute normalization statistics
train.py                 Training loop
test.py                  Evaluate a checkpoint against held-out .zarr test cases
utils.py                 Config-derived sizing, scaling factors, L2 metrics
loss.py                  Surface loss (point-wise + area-weighted)
cuml_knn_patch.py        GPU-only: works around a RAPIDS/physicsnemo cuML API mismatch -- see "Moving to a real NVIDIA GPU"
notebooks/data_distribution_analysis.ipynb   Inspect BC parameter and mesh-size distributions per train/val/test split
scripts/make_synthetic_zarr.py   Generate tiny fake .zarr cases (demo / smoke test)
scripts/rebase_surface_temperature.py   Rewrite .zarr cases to a delta-temperature target -- see "If relative-L2 looks good but predicted-vs-true shows no diagonal trend"
tests/run_smoke_test.sh  End-to-end pipeline smoke test (see above)
devtools/dali_stub/      Local-only import shim -- see "Environment notes"
```
