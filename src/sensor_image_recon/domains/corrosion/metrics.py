from __future__ import annotations

import numpy as np


def corrosion_score(image: np.ndarray) -> float:
    """Return red-channel corrosion score on a 0-100 scale from a raw [0, 1] image."""
    return float(np.asarray(image, dtype=np.float32).mean()) * 100.0


def mean_absolute_corrosion_error(real: np.ndarray, generated: np.ndarray) -> float:
    return abs(corrosion_score(real) - corrosion_score(generated))
