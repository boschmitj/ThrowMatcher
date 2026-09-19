#!/usr/bin/env python3
"""
Extract matching features from raw_mocap.csv / raw_league.csv produced by
create_throw_representation.py.

Each row's trajectory_json (PoR-relative, frame 0 = PoR, see
ThrowRepresentation.sampling_rate_hz for the native sample rate) is sampled
at its nearest native point for a common set of elapsed times after release
(COMMON_TIME_STEPS_MS) and reduced to a flat feature vector: release
kinematics, short-term velocity/direction evolution, PoR-relative
displacement, and acceleration summary stats. Time steps past the Mocap
free-flight cutoff (wall hit / unidentified ball) or past League's last
recorded point come back as None rather than being interpolated/invented.
This module writes raw feature values. The retrieval models normalize them
against the League feature distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# The 25 ms grid covers both phases of 20 Hz League samples. Values after a
# trajectory's free-flight cutoff remain missing.
COMMON_TIME_STEPS_MS: List[int] = list(range(0, 1001, 25))


def _sample_at_times(
    frame_rel: np.ndarray,
    sampling_rate_hz: float,
    values: np.ndarray,
    target_times_ms: List[int],
    max_gap_ms: float = 5.0,
    elapsed_times_ms: Optional[np.ndarray] = None,
) -> List[Optional[float]]:
    """Look up ``values`` at the native sample nearest each target elapsed time.

    Each source keeps its own sampling rate (Mocap 300 Hz, League 20 Hz); the
    native sample nearest ``t`` is used directly rather than interpolating,
    so League points always land on their own recorded values and Mocap is
    not upsampled/invented beyond what was actually tracked. ``max_gap_ms``
    (just over half a League sample period) rejects a target time once the
    nearest available sample is too far away - this is what makes times past
    the Mocap free-flight cutoff (wall hit / unidentified ball) or past
    League's last recorded point correctly come back as None.
    """
    if len(frame_rel) == 0 or sampling_rate_hz <= 0:
        return [None] * len(target_times_ms)

    if elapsed_times_ms is not None and len(elapsed_times_ms) == len(frame_rel):
        t_ms = elapsed_times_ms.astype(np.float64)
    else:
        t_ms = frame_rel.astype(np.float64) * (1000.0 / sampling_rate_hz)
    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    values = values[order]

    valid = ~np.isnan(values)
    if not valid.any():
        return [None] * len(target_times_ms)
    t_valid = t_ms[valid]
    v_valid = values[valid]

    out: List[Optional[float]] = []
    for t in target_times_ms:
        idx = int(np.argmin(np.abs(t_valid - t)))
        if abs(t_valid[idx] - t) > max_gap_ms:
            out.append(None)
        else:
            out.append(float(v_valid[idx]))
    return out


def _json_field(point: Dict[str, Any], key: str, default: float = np.nan) -> float:
    val = point.get(key, default)
    return default if val is None else val


def extract_features_from_row(row: pd.Series) -> Dict[str, Any]:
    """Compute a flat feature dict for a single raw_mocap.csv / raw_league.csv row."""
    trajectory = json.loads(row.get("trajectory_json", "[]") or "[]")
    sampling_rate_hz = float(row.get("sampling_rate_hz", 0.0) or 0.0)

    frame_rel = np.array([p.get("frame", i) for i, p in enumerate(trajectory)], dtype=np.float64)
    explicit_times = np.array([_json_field(p, "t_since_ms") for p in trajectory], dtype=np.float64)
    elapsed_times_ms = explicit_times if len(explicit_times) and np.all(np.isfinite(explicit_times)) else None
    xs = np.array([_json_field(p, "x") for p in trajectory], dtype=np.float64)
    ys = np.array([_json_field(p, "y") for p in trajectory], dtype=np.float64)
    zs = np.array([_json_field(p, "z") for p in trajectory], dtype=np.float64)
    vs = np.array([_json_field(p, "v") for p in trajectory], dtype=np.float64)
    accs = np.array([_json_field(p, "a") for p in trajectory], dtype=np.float64)
    dirs = np.array([_json_field(p, "dir") for p in trajectory], dtype=np.float64)
    vxs = np.array([_json_field(p, "vx") for p in trajectory], dtype=np.float64)
    vys = np.array([_json_field(p, "vy") for p in trajectory], dtype=np.float64)
    vzs = np.array([_json_field(p, "vz") for p in trajectory], dtype=np.float64)
    vert_angles = np.array([_json_field(p, "vert_angle") for p in trajectory], dtype=np.float64)

    features: Dict[str, Any] = {
        "throw_id": row.get("throw_id"),
        "source": row.get("source"),
        # Retrieval metadata (never selected as numeric distance features).
        "throw_type": row.get("throw_type"),
        "is_bounce": row.get("is_bounce", row.get("bounce_detected")),
        # Primary release features
        "release_speed_m_s": row.get("release_speed_m_s"),
        "release_direction_deg": row.get("release_direction_deg"),
        "release_height_m": row.get("release_height_m"),
        # Secondary / spatial features
        "por_x_m": row.get("por_x_m"),
        "por_y_m": row.get("por_y_m"),
        "por_z_m": row.get("por_z_m"),
        "trajectory_point_count": row.get("trajectory_point_count"),
        "trajectory_duration_ms": row.get("trajectory_duration_ms"),
        "sampling_rate_hz": sampling_rate_hz,
        "release_timing_type": row.get("release_timing_type", "native"),
        "first_projectile_offset_ms": row.get("first_projectile_offset_ms"),
    }

    sample_args = (frame_rel, sampling_rate_hz)
    # Do not let a League point fill the opposite 25 ms phase. Mocap has a
    # native point within 1.67 ms of every target on this grid.
    sample_kwargs = {"elapsed_times_ms": elapsed_times_ms, "max_gap_ms": 5.0}
    v_at = _sample_at_times(*sample_args, vs, COMMON_TIME_STEPS_MS, **sample_kwargs)
    d_at = _sample_at_times(*sample_args, dirs, COMMON_TIME_STEPS_MS, **sample_kwargs)
    x_at = _sample_at_times(*sample_args, xs, COMMON_TIME_STEPS_MS, **sample_kwargs)
    y_at = _sample_at_times(*sample_args, ys, COMMON_TIME_STEPS_MS, **sample_kwargs)
    z_at = _sample_at_times(*sample_args, zs, COMMON_TIME_STEPS_MS, **sample_kwargs)
    vx_at = _sample_at_times(*sample_args, vxs, COMMON_TIME_STEPS_MS, **sample_kwargs)
    vy_at = _sample_at_times(*sample_args, vys, COMMON_TIME_STEPS_MS, **sample_kwargs)
    vz_at = _sample_at_times(*sample_args, vzs, COMMON_TIME_STEPS_MS, **sample_kwargs)
    vert_angle_at = _sample_at_times(*sample_args, vert_angles, COMMON_TIME_STEPS_MS, **sample_kwargs)

    # Velocity/direction evolution + PoR-relative trajectory/displacement,
    # sampled at common elapsed times so Mocap (300 Hz) and League (20 Hz)
    # throws become directly comparable.
    for t, v, d, x, y, z, vx, vy, vz, vert_angle in zip(
        COMMON_TIME_STEPS_MS, v_at, d_at, x_at, y_at, z_at, vx_at, vy_at, vz_at, vert_angle_at
    ):
        features[f"velocity_t{t}ms"] = v
        features[f"direction_t{t}ms"] = d
        features[f"disp_x_t{t}ms"] = x
        features[f"disp_y_t{t}ms"] = y
        features[f"disp_z_t{t}ms"] = z
        features[f"velocity_x_t{t}ms"] = vx
        features[f"velocity_y_t{t}ms"] = vy
        features[f"velocity_z_t{t}ms"] = vz
        features[f"vert_angle_t{t}ms"] = vert_angle

    # Acceleration characteristics (secondary feature group)
    valid_acc = accs[~np.isnan(accs)]
    features["accel_mean_m_s2"] = float(valid_acc.mean()) if valid_acc.size else None
    features["accel_max_m_s2"] = float(valid_acc.max()) if valid_acc.size else None
    if frame_rel.size:
        idx0 = int(np.argmin(np.abs(frame_rel)))
        features["accel_at_por_m_s2"] = None if np.isnan(accs[idx0]) else float(accs[idx0])
    else:
        features["accel_at_por_m_s2"] = None

    return features


def extract_features_csv(raw_csv_path: Path, output_path: Path) -> int:
    """Read a raw_mocap.csv / raw_league.csv and write a flat features CSV.

    Returns the number of rows written.
    """
    df = pd.read_csv(raw_csv_path, sep=";")
    rows = [extract_features_from_row(row) for _, row in df.iterrows()]
    if not rows:
        output_path.write_text("")
        return 0

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract matching features from raw_mocap.csv / raw_league.csv."
    )
    parser.add_argument(
        "--raw-mocap",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "raw_mocap.csv",
    )
    parser.add_argument(
        "--raw-league",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "raw_league.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_mocap.exists():
        n = extract_features_csv(args.raw_mocap, args.output_dir / "features_mocap.csv")
        print(f"Wrote {n} rows to {args.output_dir / 'features_mocap.csv'}")
    else:
        print(f"Skipping Mocap features: {args.raw_mocap} not found")

    if args.raw_league.exists():
        n = extract_features_csv(args.raw_league, args.output_dir / "features_league.csv")
        print(f"Wrote {n} rows to {args.output_dir / 'features_league.csv'}")
    else:
        print(f"Skipping League features: {args.raw_league} not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
