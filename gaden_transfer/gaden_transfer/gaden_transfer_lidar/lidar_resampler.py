"""
Resample a ROS2 LaserScan message to exactly N uniformly-spaced rays over [0, 2π).

The training environment uses LIDAR_NUM_RAYS (72) rays at 5° spacing starting
from 0 rad (= +x axis, east).  Physical/simulated scanners typically have a
different number of beams and may start at a different angle, so we interpolate.
"""

import numpy as np
from sensor_msgs.msg import LaserScan


def resample_scan(msg: LaserScan, num_rays: int, max_range: float) -> np.ndarray:
    """Convert a LaserScan to a fixed-size, normalized distance array.

    Parameters
    ----------
    msg : LaserScan
        Incoming ROS2 laser scan.
    num_rays : int
        Target number of rays (must match training config, e.g. 72).
    max_range : float
        Maximum range used during training (e.g. 3.0 m).  Readings beyond
        this distance are clipped to 1.0 after normalisation.

    Returns
    -------
    distances : np.ndarray
        Shape (num_rays,), each value in [0, 1].  1.0 means no obstacle
        within max_range.
    """
    ranges = np.array(msg.ranges, dtype=np.float32)

    # Replace inf / nan with max_range so they normalise to 1.0
    ranges = np.where(np.isfinite(ranges), ranges, max_range)
    # Clip to [0, max_range]
    ranges = np.clip(ranges, 0.0, max_range)

    # Source angles for each beam in the original scan
    n_src = len(ranges)
    src_angles = msg.angle_min + np.arange(n_src) * msg.angle_increment  # (n_src,)

    # Wrap source angles to [0, 2π)
    src_angles = src_angles % (2.0 * np.pi)

    # Target angles: 72 uniform rays over [0, 2π)
    tgt_angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)

    # Interpolation: nearest-neighbour is fine for dense scans.
    # For sparse scans (< 72 beams) we use linear interpolation with
    # circular wrap-around.
    if n_src >= num_rays:
        # Dense → nearest neighbour on the circular angle axis
        resampled = _nearest_circular(src_angles, ranges, tgt_angles)
    else:
        # Sparse → linear interpolation with wrap-around
        resampled = _linear_circular(src_angles, ranges, tgt_angles, max_range)

    # Normalise to [0, 1]
    return np.clip(resampled / max_range, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_circular(src_angles: np.ndarray, src_ranges: np.ndarray,
                       tgt_angles: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resampling on a circular angle axis."""
    out = np.empty(len(tgt_angles), dtype=np.float32)
    for i, ta in enumerate(tgt_angles):
        diff = np.abs(src_angles - ta)
        # Circular distance
        diff = np.minimum(diff, 2.0 * np.pi - diff)
        out[i] = src_ranges[np.argmin(diff)]
    return out


def _linear_circular(src_angles: np.ndarray, src_ranges: np.ndarray,
                      tgt_angles: np.ndarray, max_range: float) -> np.ndarray:
    """Linear interpolation with circular wrap-around for sparse scans."""
    # Sort source by angle
    idx = np.argsort(src_angles)
    sa = src_angles[idx]
    sr = src_ranges[idx].astype(np.float64)

    # Duplicate first/last point for wrap-around
    sa_ext = np.concatenate([[sa[-1] - 2 * np.pi], sa, [sa[0] + 2 * np.pi]])
    sr_ext = np.concatenate([[sr[-1]], sr, [sr[0]]])

    out = np.interp(tgt_angles, sa_ext, sr_ext)
    return out.astype(np.float32)
