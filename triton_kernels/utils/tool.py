import functools
from functools import lru_cache
from collections.abc import Callable
import contextlib

import torch
from torch import Tensor
import logging
from typing import Optional, Tuple, Dict, Any

import triton
import inspect
import warnings

supports_autotune_cache = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": True} if supports_autotune_cache else {}

def _cpu_device_warning():
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)

@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'

def map_triton_backend_to_torch_device() -> str:
    backend = get_available_device()        # 'cuda' | 'hip' | 'xpu' | 'cpu' | ...
    return {'cuda': 'cuda', 'hip': 'cuda', 'xpu': 'xpu'}.get(backend, backend)


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = get_available_device()
device_name = map_triton_backend_to_torch_device()

logger = logging.getLogger(__name__)

def get_abs_err(x: Tensor, y: Tensor):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x: Tensor, y: Tensor):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def tensor_cache(
        fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[tuple] = None
    last_kwargs: Optional[dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wraps(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result
        if last_args is not None and last_kwargs is not None:
            if all(a is b for a, b in zip(args, last_args, strict=True)) and \
                all(k in last_kwargs and v is last_kwargs[k] for (k, v) in kwargs.items()):
                return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result
    return wraps

def custom_device_ctx(index: int):
    return torch.cuda.device(index)

def input_guard(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]