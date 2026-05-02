from __future__ import annotations

from typing import Tuple

import numpy as np


Coordinate = Tuple[float, float]


def haversine_km(a: Coordinate, b: Coordinate) -> float:
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    hav = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    return float(2 * 6371.0 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav)))
