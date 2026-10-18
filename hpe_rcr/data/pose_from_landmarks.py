"""Euler pose (pitch, yaw, roll degrees) from 68 landmarks via solvePnP."""

from __future__ import annotations

import numpy as np

# Canonical 3D face model (mm-ish) for a subset of stable landmarks (indices into 68-pt)
_LM_IDX = [30, 8, 36, 45, 48, 54]
_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def _wrap180(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def euler_from_landmarks(pts68: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Return pitch, yaw, roll in degrees with frontal faces near 0."""
    import cv2

    h, w = image_size
    image_pts = pts68[_LM_IDX].astype(np.float64)
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.zeros(3, dtype=np.float32)
    rmat, _ = cv2.Rodrigues(rvec)
    # OpenCV board→camera; remap so frontal 300W faces cluster near 0 (see DEVIATIONS D2)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy >= 1e-6:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = 0.0
    # Empirical unwrap: raw pitch sits near ±180 for frontal faces with this model
    pitch = _wrap180(pitch + 180.0)
    yaw = _wrap180(yaw)
    roll = _wrap180(roll)
    return np.array([pitch, yaw, roll], dtype=np.float32)


def parse_pts(path) -> np.ndarray:
    coords = []
    for ln in open(path):
        parts = ln.strip().split()
        if len(parts) == 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    arr = np.asarray(coords, dtype=np.float64)
    if arr.shape != (68, 2):
        raise ValueError(f"Expected 68 pts in {path}, got {arr.shape}")
    return arr
