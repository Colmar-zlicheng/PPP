from __future__ import annotations

import contextlib

import torch.backends.cuda as _cuda_backends

_EPS = 1e-8

_real_sdp_kernel = _cuda_backends.sdp_kernel


@contextlib.contextmanager
def _forced_math_sdp_kernel(*args, **kwargs):
    with _real_sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        yield


@contextlib.contextmanager
def force_math_sdp():
    _cuda_backends.sdp_kernel = _forced_math_sdp_kernel
    try:
        with _real_sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield
    finally:
        _cuda_backends.sdp_kernel = _real_sdp_kernel


try:
    import torch.backends.mha as _mha_backend
except ImportError:  # older torch without this backend module
    _mha_backend = None


@contextlib.contextmanager
def force_no_mha_fastpath():
    if _mha_backend is None:
        yield
        return
    prev = _mha_backend.get_fastpath_enabled()
    _mha_backend.set_fastpath_enabled(False)
    try:
        yield
    finally:
        _mha_backend.set_fastpath_enabled(prev)


@contextlib.contextmanager
def force_double_backward_safe():
    with force_math_sdp(), force_no_mha_fastpath():
        yield


