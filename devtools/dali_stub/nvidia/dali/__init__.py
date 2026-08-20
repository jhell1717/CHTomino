"""Minimal stub of the ``nvidia.dali`` package.

Only exists so that ``physicsnemo.datapipes.cae`` (an NVIDIA DALI/CUDA-only
module unrelated to the DoMINO/zarr code path this project uses) can be
imported on machines without DALI installed (e.g. this macOS/CPU dev box).
Nothing here is ever actually called -- if it is, it raises loudly.
On the real training cluster (Linux + NVIDIA GPU), install the real
``nvidia-dali`` package and this stub is never used.
"""


class _Unavailable:
    def __getattr__(self, item):
        raise ImportError(
            "nvidia.dali is stubbed out for CPU-only local testing; "
            "install the real nvidia-dali package to use it."
        )

    def __call__(self, *a, **kw):
        raise ImportError(
            "nvidia.dali is stubbed out for CPU-only local testing; "
            "install the real nvidia-dali package to use it."
        )


class Pipeline:
    def __init__(self, *a, **kw):
        raise ImportError(
            "nvidia.dali is stubbed out for CPU-only local testing; "
            "install the real nvidia-dali package to use it."
        )


class types:
    class SampleInfo:
        pass


fn = _Unavailable()
pipeline_def = _Unavailable()
