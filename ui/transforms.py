"""Named numeric transforms for widget ↔ backend value mapping.

Each entry is (forward, inverse_or_None). `forward` maps UI → backend (used
when collecting configs to send downstream). `inverse` maps backend → UI
(used when initializing widget values from a meta.yaml default). When the
transform is not invertible (e.g. abs), `inverse` is None and callers must
provide an explicit fallback.
"""
from __future__ import annotations

from typing import Callable


_TRANSFORMS: dict[str, tuple[Callable[[float], float], Callable[[float], float] | None]] = {
    "identity":   (lambda x: x,      lambda x: x),
    "negate":     (lambda x: -x,     lambda x: -x),
    "complement": (lambda x: 1 - x,  lambda x: 1 - x),
    "abs":        (lambda x: abs(x), None),  # not invertible (sign ambiguous)
}


def apply_forward(name: str | None, value):
    if name is None or value is None:
        return value
    entry = _TRANSFORMS.get(name)
    if entry is None:
        return value
    try:
        return entry[0](value)
    except Exception:
        return value


def apply_inverse(name: str | None, value):
    """Return inverse-transformed value, or None if transform is not invertible.

    Returns the original value unchanged when name is None / unknown.
    """
    if name is None or value is None:
        return value
    entry = _TRANSFORMS.get(name)
    if entry is None:
        return value
    inv = entry[1]
    if inv is None:
        return None
    try:
        return inv(value)
    except Exception:
        return None
