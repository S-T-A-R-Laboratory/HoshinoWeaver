import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_uint8_image(rng):
    def _make(h=64, w=64, channels=3):
        return rng.integers(0, 256, size=(h, w, channels), dtype=np.uint8)
    return _make


@pytest.fixture
def synthetic_uint16_image(rng):
    def _make(h=64, w=64, channels=3):
        return rng.integers(0, 65536, size=(h, w, channels), dtype=np.uint16)
    return _make


@pytest.fixture
def synthetic_float_image(rng):
    def _make(h=64, w=64, channels=3, bg=0.2, noise_std=0.02):
        return (bg + rng.standard_normal((h, w, channels)) * noise_std).astype(np.float32).clip(0, 1)
    return _make


@pytest.fixture
def synthetic_star_image(rng):
    """Generate an image with Gaussian point sources on a uniform background."""
    def _make(h=128, w=128, n_stars=5, bg=0.1, star_peak=0.9, star_sigma=3.0):
        img = np.full((h, w), bg, dtype=np.float32)
        yy, xx = np.mgrid[:h, :w]
        star_positions = []
        for _ in range(n_stars):
            cy = rng.integers(star_sigma * 3, h - star_sigma * 3)
            cx = rng.integers(star_sigma * 3, w - star_sigma * 3)
            g = star_peak * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * star_sigma**2))
            img += g
            star_positions.append((cx, cy))
        return img.clip(0, 1), star_positions
    return _make
