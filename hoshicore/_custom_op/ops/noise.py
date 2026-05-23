"""Noise equalization custom-op runtime backends."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any, Callable

import numpy as np

from hoshicore._custom_op._dispatch import apply_compiled_threads as _apply_compiled_threads
from hoshicore._custom_op._dispatch import compiled_build_info as _compiled_build_info
from hoshicore._custom_op._dispatch import debug_enabled as _debug_enabled
from hoshicore._custom_op._dispatch import debug_log
from hoshicore._custom_op._dispatch import fallback_preference as _fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result


_debug_log = partial(debug_log, "noise")


def _validate_equalize_noise_inputs(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_arr = np.asarray(max_img)
    filled_std_arr = np.asarray(filled_std_img)
    if max_arr.shape != filled_std_arr.shape:
        raise ValueError("equalize_noise_correct: shape mismatch")
    if max_arr.dtype != filled_std_arr.dtype:
        raise ValueError("equalize_noise_correct: dtype mismatch")
    if not np.issubdtype(max_arr.dtype, np.floating):
        raise ValueError("equalize_noise_correct: floating-point arrays required")
    if not max_arr.flags.c_contiguous:
        max_arr = np.ascontiguousarray(max_arr)
    if not filled_std_arr.flags.c_contiguous:
        filled_std_arr = np.ascontiguousarray(filled_std_arr)
    return max_arr, filled_std_arr


def _validate_highlight_preserve(highlight_preserve: float) -> float:
    value = float(highlight_preserve)
    if not (0.0 <= value < 1.0):
        raise ValueError("equalize_noise_correct: highlight_preserve must be in [0, 1)")
    return value


def equalize_noise_correct_numpy(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
    sigma_ref: float,
    c_n_eff: float,
    max_value: float,
    highlight_preserve: float,
) -> np.ndarray:
    max_arr, filled_std_arr = _validate_equalize_noise_inputs(max_img, filled_std_img)
    highlight_value = _validate_highlight_preserve(highlight_preserve)
    max_value_float = float(max_value)
    fix_strength = (
        (max_value_float * highlight_value - max_arr).clip(max=0)
        / (max_value_float * (1.0 - highlight_value))
        + 1.0
    )
    fixed_std_img = fix_strength * filled_std_arr
    corrected = max_arr - (fixed_std_img - float(sigma_ref)) * float(c_n_eff)
    return np.clip(corrected, a_min=0.0, a_max=max_value_float)


def equalize_noise_correct_compiled(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
    sigma_ref: float,
    c_n_eff: float,
    max_value: float,
    highlight_preserve: float,
) -> np.ndarray:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "equalize_noise_correct"):
        raise RuntimeError("compiled custom op backend is unavailable")
    max_arr, filled_std_arr = _validate_equalize_noise_inputs(max_img, filled_std_img)
    highlight_value = _validate_highlight_preserve(highlight_preserve)
    _apply_compiled_threads("equalize_noise_correct", max_arr)
    return module.equalize_noise_correct(
        max_arr,
        filled_std_arr,
        float(sigma_ref),
        float(c_n_eff),
        float(max_value),
        highlight_value,
    )


@lru_cache(maxsize=2)
def _select_equalize_noise_backend(
    preference: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray, float, float, float, float], np.ndarray]]:
    module, compiled_error = _load_compiled_module_result()
    if module is not None and hasattr(module, "equalize_noise_correct"):
        return "compiled", equalize_noise_correct_compiled

    if compiled_error:
        _debug_log(f"compiled backend unavailable, reason: {compiled_error}")

    return "numpy", equalize_noise_correct_numpy


def equalize_noise_correct(
    max_img: np.ndarray,
    filled_std_img: np.ndarray,
    sigma_ref: float,
    c_n_eff: float,
    max_value: float,
    highlight_preserve: float,
) -> np.ndarray:
    backend_name, backend = _select_equalize_noise_backend(_fallback_preference())
    if backend_name == "compiled":
        max_arr, _ = _validate_equalize_noise_inputs(max_img, filled_std_img)
        _apply_compiled_threads("equalize_noise_correct", max_arr)
    return backend(
        max_img,
        filled_std_img,
        sigma_ref,
        c_n_eff,
        max_value,
        highlight_preserve,
    )
