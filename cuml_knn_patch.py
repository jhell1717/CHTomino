"""Workaround for a RAPIDS/physicsnemo version mismatch in the cuML kNN backend.

physicsnemo 1.3.0's cuML backend for physicsnemo.utils.neighbors.knn()
(physicsnemo/utils/neighbors/knn/_cuml_impl.py) manually constructs a
``cuml.Handle(stream=...)`` to bind the kNN search to the current PyTorch
CUDA stream. Newer RAPIDS releases (physicsnemo only pins ``cuml>=24.0.0``,
no upper bound) have reorganized cuML's top-level namespace behind a lazy
``__getattr__``, which no longer exposes ``Handle`` there -- raising
``AttributeError: module cuml has no attribute Handle`` the first time
DoMINODataPipe calls knn() on a CUDA tensor.

This rebuilds physicsnemo's knn() dispatcher (same auto/torch/scipy/cuml
backend selection, same dtype/device validation) but has the cuML backend go
through ``cuml.neighbors.NearestNeighbors`` directly -- cuML's stable,
long-standing public API -- instead of a manually constructed Handle. We lose
physicsnemo's explicit stream binding, so this adds explicit
synchronization around the cuML/cupy call instead, trading a little
performance for correctness safety.

Imported by train.py / test.py / compute_statistics.py already
(`import cuml_knn_patch  # noqa: F401`, near the top of each file). It's a
no-op if cuML isn't installed/available at all, so it's harmless to import
even on a machine without a GPU (e.g. the one this project was built on).

Pinning cuml-cu12/cuml-cu11 to a 25.x point release instead (see README
"Moving to a real NVIDIA GPU") needs no code changes at all, and
physicsnemo 1.3.0 was itself built against a 25.x RAPIDS release (confirmed
via its own Dockerfile) -- so if that pin alone gets `cuml.Handle` working,
this file's patch becomes redundant (harmless either way, since it only
changes behavior when CUML_AVAILABLE and points are on CUDA) and can be
removed from the three imports above if you'd rather not carry it.

UNVERIFIED beyond static reasoning about cuML's public API and the
traceback that motivated it -- there was no GPU/cuML available in the
environment this project was built in. If it still doesn't work, run:
    python -c "import cuml; print(cuml.__version__)"
    python -c "import cuml; print([n for n in dir(cuml) if 'andle' in n.lower()])"
    python -c "import cuml.common.handle as h; print(h.Handle)"
and report back what each one prints/errors with, so this can be corrected
against your actual installed version instead of guessed at again.
"""

import torch

from physicsnemo.datapipes.cae import domino_datapipe
from physicsnemo.utils.neighbors.knn._cuml_impl import CUML_AVAILABLE
from physicsnemo.utils.neighbors.knn._scipy_impl import SCIPY_AVAILABLE
from physicsnemo.utils.neighbors.knn._scipy_impl import knn_impl as _knn_scipy
from physicsnemo.utils.neighbors.knn._torch_impl import knn_impl as _knn_torch


def _knn_cuml_no_handle(points: torch.Tensor, queries: torch.Tensor, k: int):
    """Same as physicsnemo's _cuml_impl.knn_impl, minus the broken cuml.Handle(...)."""
    import cuml
    import cupy as cp

    # physicsnemo binds cuML's work to torch's current stream via an explicit
    # Handle; without that, fall back to full synchronization around the
    # cuML/cupy call so results can't be read (or points/queries overwritten)
    # before the search actually completes on cuML's own stream.
    torch.cuda.synchronize()

    points_cp = cp.from_dlpack(points)
    queries_cp = cp.from_dlpack(queries)

    model = cuml.neighbors.NearestNeighbors(n_neighbors=k)
    model.fit(points_cp)
    distance, indices = model.kneighbors(queries_cp)

    indices_t = torch.from_dlpack(indices)
    distance_t = torch.from_dlpack(distance)
    cp.cuda.get_current_stream().synchronize()

    return indices_t, distance_t


def _patched_knn(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for physicsnemo.utils.neighbors.knn.knn."""
    if backend not in ["cuml", "torch", "scipy", "auto"]:
        raise ValueError(
            f"`knn` backend must be in ['cuml', 'torch', 'scipy', 'auto'], got {backend=}"
        )
    if points.device != queries.device:
        raise ValueError(
            f"`knn` points and queries must be on the same device, got "
            f"{points.device=} and {queries.device=}"
        )
    if points.dtype != queries.dtype:
        raise ValueError(
            f"`knn` points and queries must have the same dtype, got "
            f"{points.dtype=} and {queries.dtype=}"
        )

    if backend == "auto":
        if points.is_cuda:
            backend = "cuml" if CUML_AVAILABLE else "torch"
        else:
            backend = "scipy" if SCIPY_AVAILABLE else "torch"

    original_dtype = points.dtype
    if points.dtype == torch.bfloat16 and backend in ("cuml", "scipy"):
        points = points.to(torch.float32)
        queries = queries.to(torch.float32)

    if backend == "scipy":
        if points.device.type != "cpu":
            raise ValueError(f"`knn` scipy backend does not support CUDA, got {points.device=}")
        indices, distances = _knn_scipy(points, queries, k)
    elif backend == "cuml":
        if points.device.type != "cuda":
            raise ValueError(f"`knn` cuml backend does not support CPU, got {points.device=}")
        indices, distances = _knn_cuml_no_handle(points, queries, k)
    elif backend == "torch":
        indices, distances = _knn_torch(points, queries, k)
    else:
        raise NotImplementedError(f"Unknown backend: {backend}")

    return indices, distances.to(original_dtype)


# DoMINODataPipe calls `knn(...)` via the name it bound at its own import
# time (`from physicsnemo.utils.neighbors import knn`), so the fix has to
# overwrite that binding specifically -- patching physicsnemo.utils.neighbors
# .knn.knn (or ._cuml_impl.knn_impl) would leave domino_datapipe's already
# -bound reference to the original, broken function untouched.
domino_datapipe.knn = _patched_knn
