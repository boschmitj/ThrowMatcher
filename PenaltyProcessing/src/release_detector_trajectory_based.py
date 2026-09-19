from __future__ import annotations

import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gc
import os
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

# Module-level logger. Configured in main(); defaults to WARNING so library
# use does not spam stdout unless the caller opts in.
logger = logging.getLogger("release_detector_trajectory_based")

from ball_trajectory import BallPoint, serialize_point, serialize_trajectory
from fixture_resolution import build_edge_case_mappings, build_fixture_index, resolve_fixture_file
from penalty_time_utils import canonical_team_name, parse_penalty_local_time, parse_position_local_time, try_float, try_int


@dataclass
class ProjectileModel:
    """Timestamp-based polynomial model of the free-flight ball trajectory."""

    origin_dt: datetime
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def position_at(self, dt: datetime) -> Tuple[float, float, float]:
        t = (dt - self.origin_dt).total_seconds()
        return tuple(float(np.polynomial.polynomial.polyval(t, c)) for c in (self.x, self.y, self.z))  # type: ignore[return-value]

    def velocity_at(self, dt: datetime) -> Tuple[float, float, float]:
        t = (dt - self.origin_dt).total_seconds()
        values = [float(np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(c))) for c in (self.x, self.y, self.z)]
        return values[0], values[1], values[2]


class FixtureCache:
    def __init__(self) -> None:
        # Simple single-entry cache: fixtures are processed in contiguous blocks
        # so we only keep the currently-loaded fixture in memory. This avoids
        # accumulating many large files.
        self._current_path: Optional[Path] = None
        self._current_points: Optional[List[BallPoint]] = None

    def get_points(self, path: Path, start_dt: Optional[datetime] = None, time_window_seconds: int = 5) -> List[BallPoint]:
        # Resolve path to a stable absolute path for comparisons and caching.
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        # If requested path is already loaded, return cached points.
        if self._current_path == resolved and self._current_points is not None and start_dt is None:
            return self._current_points

        # Load via pandas-backed loader (memory efficient for column selection)
        pts = _build_ball_points_from_file(path, start_dt=start_dt, time_window_seconds=time_window_seconds)
        # Replace current cache entry
        # Store resolved path in cache
        self._current_path = resolved
        self._current_points = pts
        # Trigger GC to free previous large DataFrames if any
        gc.collect()
        return pts

    def prefetch(self, path: Path, start_dt: Optional[datetime] = None, time_window_seconds: int = 5) -> None:
        try:
            _ = self.get_points(path, start_dt=start_dt, time_window_seconds=time_window_seconds)
        except Exception:
            pass


def _coerce_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_id(value: Any) -> str:
    text = _coerce_text(value)
    if not text:
        return ""
    return text.replace(".0", "")


def _build_ball_points_from_rows(rows: Sequence[Dict[str, str]], start_dt: Optional[datetime] = None) -> List[BallPoint]:
    points: List[BallPoint] = []
    for row in rows:
        if (row.get("group name") or "").strip() != "Ball":
            continue
        local_dt = parse_position_local_time(_coerce_text(row.get("formatted local time")))
        x = try_float(row.get("x in m", ""))
        y = try_float(row.get("y in m", ""))
        z = try_float(row.get("z in m", ""))
        if local_dt is None or x is None or y is None or z is None:
            continue
        if start_dt is not None and local_dt < start_dt - timedelta(seconds=5):
            continue
        if start_dt is not None and local_dt > start_dt + timedelta(seconds=5):
            continue
        points.append(
            BallPoint(
                local_dt=local_dt,
                ts_ms=try_int(row.get("ts in ms", "")),
                x=x,
                y=y,
                z=z,
                speed=try_float(row.get("speed in m/s", "")) or float("nan"),
                accel=try_float(row.get("acceleration in m/s2", "")) or float("nan"),
                direction=try_float(row.get("direction of movement in deg", "")),
            )
        )
    points.sort(key=lambda p: (p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


def _load_fixture_rows(positions_file: Path) -> List[Dict[str, str]]:
    with positions_file.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _build_ball_points_from_file(positions_file: Path, start_dt: Optional[datetime] = None, time_window_seconds: int = 5) -> List[BallPoint]:
    # Try a pandas-backed load for performance and column selection. If pandas
    # fails for any reason, fall back to the csv-based streaming loader.
    usecols = [
        "group name",
        "formatted local time",
        "x in m",
        "y in m",
        "z in m",
        "ts in ms",
        "speed in m/s",
        "acceleration in m/s2",
        "direction of movement in deg",
    ]
    try:
        df = pd.read_csv(positions_file, delimiter=";", usecols=usecols, dtype=str, low_memory=True)
    except Exception:
        return _build_ball_points_from_rows(_load_fixture_rows(positions_file), start_dt=start_dt)

    # Keep only ball rows
    if "group name" in df.columns:
        df = df[df["group name"].str.strip() == "Ball"]
    if df.empty:
        return []

    # Parse datetime strings into python datetimes using existing parser
    df["local_dt"] = df["formatted local time"].map(lambda s: parse_position_local_time(_coerce_text(s)))

    # Convert numeric columns
    for col in ["x in m", "y in m", "z in m", "ts in ms", "speed in m/s", "acceleration in m/s2", "direction of movement in deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    if "local_dt" not in df.columns:
        return []
    df = df[df["local_dt"].notna() & df["x in m"].notna() & df["y in m"].notna() & df["z in m"].notna()]
    if df.empty:
        return []

    if start_dt is not None:
        before = start_dt - timedelta(seconds=time_window_seconds)
        after = start_dt + timedelta(seconds=time_window_seconds)
        df = df[(df["local_dt"] >= before) & (df["local_dt"] <= after)]
        if df.empty:
            return []

    df = df.sort_values(["local_dt", "ts in ms"], ascending=[True, True])

    points: List[BallPoint] = []
    for _, row in df.iterrows():
        ts_val = row.get("ts in ms")
        ts_ms = int(ts_val) if not (pd.isna(ts_val)) else None
        speed = float(row["speed in m/s"]) if not pd.isna(row.get("speed in m/s")) else float("nan")
        accel = float(row["acceleration in m/s2"]) if not pd.isna(row.get("acceleration in m/s2")) else float("nan")
        direction = float(row["direction of movement in deg"]) if not pd.isna(row.get("direction of movement in deg")) else None
        points.append(
            BallPoint(
                local_dt=row["local_dt"],
                ts_ms=ts_ms,
                x=float(row["x in m"]),
                y=float(row["y in m"]),
                z=float(row["z in m"]),
                speed=speed,
                accel=accel,
                direction=direction,
            )
        )

    return points


def _filter_points_for_penalty(rows: Sequence[Dict[str, str]], start_dt: Optional[datetime] = None) -> List[BallPoint]:
    # Backwards-compatible wrapper: if the caller passes a Path-like object,
    # allow streaming directly from file. Otherwise assume pre-read rows.
    if isinstance(rows, (str, Path)):
        return _build_ball_points_from_file(Path(rows), start_dt=start_dt)
    return _build_ball_points_from_rows(rows, start_dt=start_dt)


def _find_window(points: Sequence[BallPoint], start_dt: datetime) -> Tuple[int, int]:
    start_idx = None
    for idx, point in enumerate(points):
        if start_idx is None and point.local_dt >= start_dt - timedelta(seconds=1):
            start_idx = idx
            break
    if start_idx is None:
        start_idx = 0

    end_idx = None
    for idx in range(start_idx, len(points)):
        if abs(points[idx].x) > 20.0:
            end_idx = min(len(points) - 1, idx + 5)
            break
    if end_idx is None:
        end_idx = len(points) - 1
    return start_idx, end_idx


def _fit_reference_line(points: Sequence[BallPoint]) -> Optional[Tuple[float, float, float]]:
    reference_points = [p for p in points if 14.0 < abs(p.x) < 16.0]
    if len(reference_points) < 3:
        return None

    reference_points = reference_points[:5]
    xs = [p.x for p in reference_points]
    ys = [p.y for p in reference_points]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in residuals) / max(n, 1))
    tolerance = max(0.07, 2.5 * rmse + 0.03)
    return slope, intercept, tolerance


def _has_strong_direction_change(
    points: Sequence[BallPoint],
    idx: int,
    reference_line: Optional[Tuple[float, float, float]] = None,
) -> bool:
    if idx < 2 or not 16.0 < abs(points[idx].x) < 20.0:
        return False

    point = points[idx]
    prev_prev = points[idx - 2]
    prev = points[idx - 1]
    curr = points[idx]

    prev_dt = (prev.local_dt - prev_prev.local_dt).total_seconds()
    curr_dt = (curr.local_dt - prev.local_dt).total_seconds()
    if prev_dt <= 0.0 or curr_dt <= 0.0:
        return False
    prev_vel = (prev.x - prev_prev.x) / prev_dt
    curr_vel = (curr.x - prev.x) / curr_dt
    if prev_vel * curr_vel < 0.0 and abs(prev_vel) > 3.0 and abs(curr_vel) > 3.0:
        if idx + 1 < len(points):
            next_dt = (points[idx + 1].local_dt - curr.local_dt).total_seconds()
            next_vel = (points[idx + 1].x - curr.x) / next_dt if next_dt > 0.0 else 0.0
            if abs(next_vel) > 3.0 and next_vel * curr_vel > 0.0:
                return True
        return True

    if reference_line is not None:
        slope, intercept, tolerance = reference_line
        residual = point.y - (slope * point.x + intercept)
        if abs(residual) > tolerance:
            return True

    return False


def _rotate_180_z(points: Sequence[BallPoint]) -> List[BallPoint]:
    """Rotate points by 180 degrees around the Z axis.

    Applies ``newX = -oldX``, ``newY = -oldY``, ``newZ = oldZ`` to every point.
    Also rotates the ``direction`` heading by 180° so the direction stays
    consistent with the rotated coordinate frame (e.g. a throw toward -x
    becomes a throw toward +x, so 180° becomes 0°).
    This maps a throw performed on the left side of the field (-x) onto the
    right side (+x) while preserving right-handedness.
    """
    rotated: List[BallPoint] = []
    for point in points:
        rotated.append(
            BallPoint(
                local_dt=point.local_dt,
                ts_ms=point.ts_ms,
                x=-point.x,
                y=-point.y,
                z=point.z,
                speed=point.speed,
                accel=point.accel,
                direction=(point.direction + 180.0) % 360.0 if point.direction is not None else None,
            )
        )
    return rotated


def _is_left_side(points: Sequence[BallPoint]) -> bool:
    """Return True if the throw is performed on the left side of the field (-x).

    Determines the side from the median x-coordinate of the points near the
    release area (|x| < 14 m, i.e. before the goal line).
    """
    near_points = [p.x for p in points if abs(p.x) > 12.0]
    if not near_points:
        return False
    median_x = float(np.median(near_points))
    return median_x < 0.0


def _has_deflection(points: Sequence[BallPoint], start_idx: int, end_idx: int) -> bool:
    """Detect whether the trajectory contains a deflection (e.g. by the goalkeeper).

    Uses the same strong-direction-change heuristic as the release detector:
    a deflection is present if any point after the release area shows a strong
    direction change or deviates from the fitted reference line.
    """
    if len(points) < 5:
        return False

    # Fit the undisturbed approach line before the goalkeeper-contact zone.
    # Only samples in 16 < |x| < 20 can be classified as keeper deflections.
    reference_line = _fit_reference_line(points[start_idx:end_idx + 1])
    bounce_idx = find_bounce_idx(points, start_idx, end_idx)
    scan_end = min(end_idx, bounce_idx - 1) if bounce_idx is not None else end_idx
    for idx in range(max(start_idx, 2), scan_end + 1):
        if not 16.0 < abs(points[idx].x) < 20.0:
            continue
        if _has_strong_direction_change(points, idx, reference_line=reference_line):
            return True
        if idx >= 3:
            a, b, c = points[idx - 2], points[idx - 1], points[idx]
            dt1 = (b.local_dt - a.local_dt).total_seconds()
            dt2 = (c.local_dt - b.local_dt).total_seconds()
            if dt1 > 0.0 and dt2 > 0.0:
                v1 = np.array([(b.x-a.x)/dt1, (b.y-a.y)/dt1, (b.z-a.z)/dt1])
                v2 = np.array([(c.x-b.x)/dt2, (c.y-b.y)/dt2, (c.z-b.z)/dt2])
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if n1 > 2.0 and n2 > 2.0:
                    angle = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0))))
                    if angle > 28.0 and min(a.z, b.z, c.z) > 0.25:
                        return True
    return False


def find_bounce_idx(points: Sequence[BallPoint], start_idx: int, end_idx: int) -> Optional[int]:
    """Return the last ground-contact sample in the 14 < |x| < 20 zone."""
    if len(points) < 3:
        return None
    lo, hi = max(1, start_idx), min(end_idx, len(points) - 2)
    for idx in range(hi, lo - 1, -1):
        p0, p1, p2 = points[idx - 1], points[idx], points[idx + 1]
        if not (14.0 < abs(p1.x) < 20.0) or p1.z > 0.30:
            continue
        dt0 = (p1.local_dt - p0.local_dt).total_seconds()
        dt1 = (p2.local_dt - p1.local_dt).total_seconds()
        if dt0 <= 0.0 or dt1 <= 0.0:
            continue
        vz_before = (p1.z - p0.z) / dt0
        vz_after = (p2.z - p1.z) / dt1
        if p1.z <= p0.z and p1.z <= p2.z and vz_before < -0.5 and vz_after > 0.5:
            return idx
    return None


def _fit_projectile_model(points: Sequence[BallPoint]) -> Optional[ProjectileModel]:
    if len(points) < 3:
        return None
    origin = points[0].local_dt
    times = np.asarray([(p.local_dt - origin).total_seconds() for p in points], dtype=float)
    if len(np.unique(times)) < 3 or times[-1] <= 0.0:
        return None
    try:
        x = np.polynomial.polynomial.polyfit(times, [p.x for p in points], 1)
        y = np.polynomial.polynomial.polyfit(times, [p.y for p in points], 1)
        z = np.polynomial.polynomial.polyfit(times, [p.z for p in points], 2)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not all(np.all(np.isfinite(c)) for c in (x, y, z)):
        return None
    return ProjectileModel(origin, x, y, z)


def _robust_scale(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return 0.0
    median = float(np.median(array))
    return 1.4826 * float(np.median(np.abs(array - median)))


def _anchor_tolerances(points: Sequence[BallPoint], model: ProjectileModel) -> Dict[str, float]:
    residuals = []
    direction = np.asarray((model.x[1], model.y[1]), dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    for point in points:
        predicted = model.position_at(point.local_dt)
        xyz = np.asarray((point.x, point.y, point.z)) - np.asarray(predicted)
        if direction_norm > 1e-9:
            xy_line = abs(direction[0] * (point.y - model.y[0]) - direction[1] * (point.x - model.x[0])) / direction_norm
        else:
            xy_line = math.hypot(xyz[0], xyz[1])
        residuals.append((abs(xyz[0]), abs(xyz[1]), abs(xyz[2]), xy_line))
    columns = list(zip(*residuals))
    minimums = (0.18, 0.15, 0.25, 0.15)
    names = ("x", "y", "z", "xy_line")
    return {name: max(minimum, 4.0 * _robust_scale(column)) for name, minimum, column in zip(names, minimums, columns)}


def _nearby_interval_statistics(points: Sequence[BallPoint]) -> Tuple[Optional[float], Optional[float]]:
    velocities: List[float] = []
    displacements: List[float] = []
    for first, second in zip(points[:3], points[1:4]):
        dt = (second.local_dt - first.local_dt).total_seconds()
        if dt <= 0.0:
            continue
        dx = abs(second.x - first.x)
        displacements.append(dx)
        velocities.append(dx / dt)
    return (
        float(np.median(velocities)) if velocities else None,
        float(np.median(displacements)) if displacements else None,
    )


def _has_vertical_turn_in_release_zone(
    points: Sequence[BallPoint],
    start_idx: int,
    end_idx: int,
    *,
    goal_x: float = 20.0,
    min_goal_distance: float = 6.0,
    max_goal_distance: float = 7.5,
    noise_velocity_m_s: float = 0.35,
    noise_displacement_m: float = 0.025,
) -> bool:
    """Detect a meaningful reversal of vertical motion near release.

    The backward projectile search can otherwise absorb hand-carried samples
    when x/y are linear and z happens to look parabolic.  Only intervals whose
    endpoints lie in the configured release-distance band are considered.
    Tiny z changes are ignored, both in metres and metres/second, so tracking
    jitter cannot create a false turn.
    """
    lo = max(0, start_idx)
    hi = min(end_idx, len(points) - 1)
    previous_direction = 0
    for first, second in zip(points[lo:hi], points[lo + 1:hi + 1]):
        first_distance = goal_x - abs(first.x)
        second_distance = goal_x - abs(second.x)
        if not (
            min_goal_distance <= first_distance <= max_goal_distance
            and min_goal_distance <= second_distance <= max_goal_distance
        ):
            continue
        dt = (second.local_dt - first.local_dt).total_seconds()
        dz = second.z - first.z
        if dt <= 0.0 or abs(dz) <= noise_displacement_m or abs(dz / dt) <= noise_velocity_m_s:
            continue
        direction = 1 if dz > 0.0 else -1
        if previous_direction and direction != previous_direction:
            return True
        previous_direction = direction
    return False


def _evaluate_projectile_candidate(
    point: BallPoint,
    next_point: BallPoint,
    anchor_points: Sequence[BallPoint],
    model: ProjectileModel,
    tolerances: Dict[str, float],
) -> Dict[str, Any]:
    predicted = model.position_at(point.local_dt)
    x_residual = abs(point.x - predicted[0])
    y_residual = abs(point.y - predicted[1])
    z_residual = abs(point.z - predicted[2])
    direction = np.asarray((model.x[1], model.y[1]), dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    xy_line_residual = (
        abs(direction[0] * (point.y - model.y[0]) - direction[1] * (point.x - model.x[0])) / direction_norm
        if direction_norm > 1e-9 else math.hypot(x_residual, y_residual)
    )

    dt = (next_point.local_dt - point.local_dt).total_seconds()
    candidate_speed = abs(next_point.x - point.x) / dt if dt > 0.0 else None
    clean_speed, clean_dx = _nearby_interval_statistics(anchor_points)
    dx = abs(next_point.x - point.x)
    dx_ratio = dx / clean_dx if clean_dx is not None and clean_dx > 1e-9 else None
    velocity_ratio = candidate_speed / clean_speed if candidate_speed is not None and clean_speed and clean_speed > 1e-9 else None

    normalized = {
        "x": x_residual / tolerances["x"],
        "y": y_residual / tolerances["y"],
        "z": z_residual / tolerances["z"],
        "xy_line": xy_line_residual / tolerances["xy_line"],
    }
    primary_geometry_ok = normalized["x"] <= 1.0 and normalized["y"] <= 1.0 and normalized["xy_line"] <= 1.0
    vertical_ok = normalized["z"] <= 1.5
    gross_motion_mismatch = (
        (dx_ratio is not None and dx_ratio < 0.22)
        or (velocity_ratio is not None and velocity_ratio < 0.25)
        or dt <= 0.0
    )
    compatible = primary_geometry_ok and vertical_ok and not gross_motion_mismatch
    worst_geometry = max(normalized.values())
    extreme = worst_geometry > 3.0 or (gross_motion_mismatch and worst_geometry > 1.5)
    return {
        "compatible": compatible,
        "extreme": extreme,
        "x_residual": x_residual,
        "y_residual": y_residual,
        "z_residual": z_residual,
        "xy_line_residual": xy_line_residual,
        "dx_ratio": dx_ratio,
        "velocity_ratio": velocity_ratio,
        "normalized_error": float(np.mean(list(normalized.values()))),
    }


def detect_backward_projectile_segment(
    points: Sequence[BallPoint],
    start_idx: int,
    end_idx: int,
    *,
    reject_vertical_turn_near_release: bool = True,
    release_zone_goal_x: float = 20.0,
    release_zone_min_goal_distance: float = 6.0,
    release_zone_max_goal_distance: float = 7.5,
    vertical_noise_velocity_m_s: float = 0.35,
    vertical_noise_displacement_m: float = 0.025,
) -> Optional[Dict[str, Any]]:
    """Fit the clean trajectory tail and grow the projectile segment backward."""
    if len(points) < 6:
        return None
    start_idx, end_idx = max(0, start_idx), min(end_idx, len(points) - 1)
    projectile_end_idx = end_idx
    for idx in range(start_idx, end_idx + 1):
        if abs(points[idx].x) >= 20.0:
            projectile_end_idx = max(start_idx, idx - 1)
            break
    under_14 = [i for i in range(start_idx, projectile_end_idx + 1) if abs(points[i].x) < 14.0]
    search_start = under_14[-1] + 1 if under_14 else start_idx
    bounce_idx = find_bounce_idx(points, search_start, projectile_end_idx)
    if bounce_idx is not None:
        projectile_end_idx = bounce_idx - 1

    # The post-gate samples are the clearly valid seed. Never cross back below
    # 14 m merely to meet a sample-count or duration target: earlier samples
    # are admitted only by the compatibility test below.
    seed_start = search_start
    if projectile_end_idx - seed_start + 1 < 3:
        return None
    anchor_points = points[seed_start:projectile_end_idx + 1]
    anchor_point_count = len(anchor_points)
    anchor_low_confidence = anchor_point_count == 3
    accepted_start = seed_start
    anchor_model = _fit_projectile_model(anchor_points)
    if anchor_model is None:
        return None
    tolerances = _anchor_tolerances(anchor_points, anchor_model)
    pending_failures: List[Tuple[int, Dict[str, Any]]] = []
    compatible_after_failure = 0
    boundary_diagnostics: Optional[Dict[str, Any]] = None
    for idx in range(seed_start - 1, start_idx - 1, -1):
        diagnostics = _evaluate_projectile_candidate(
            points[idx], points[idx + 1], anchor_points, anchor_model, tolerances,
        )
        vertical_turn = reject_vertical_turn_near_release and _has_vertical_turn_in_release_zone(
            points,
            idx,
            seed_start,
            goal_x=release_zone_goal_x,
            min_goal_distance=release_zone_min_goal_distance,
            max_goal_distance=release_zone_max_goal_distance,
            noise_velocity_m_s=vertical_noise_velocity_m_s,
            noise_displacement_m=vertical_noise_displacement_m,
        )
        diagnostics["vertical_turn_near_release"] = vertical_turn
        if vertical_turn:
            # This is physical boundary evidence rather than an isolated bad
            # measurement: stop before hand motion can enter the flight segment.
            diagnostics["compatible"] = False
            diagnostics["extreme"] = True
        if diagnostics["compatible"]:
            # Treat a single intervening failure as measurement noise. The
            # fixed anchor remains unchanged, so it cannot contaminate later tests.
            accepted_start = idx
            if pending_failures:
                compatible_after_failure += 1
                if compatible_after_failure >= 2:
                    pending_failures.clear()
                    compatible_after_failure = 0
            continue
        pending_failures.append((idx, diagnostics))
        if diagnostics["extreme"]:
            boundary_diagnostics = pending_failures[0][1]
            if len(pending_failures) > 1:
                accepted_start = pending_failures[0][0] + 1
            break
        if len(pending_failures) >= 2:
            # The first failure encountered is immediately adjacent to the
            # accepted projectile region and therefore defines the boundary.
            boundary_diagnostics = pending_failures[0][1]
            accepted_start = pending_failures[0][0] + 1
            break
    else:
        # An unconfirmed single failure is noise, not a transition.
        if len(pending_failures) == 1:
            accepted_start = pending_failures[0][0]
    if accepted_start <= start_idx:
        return None
    final_model = _fit_projectile_model(points[accepted_start:projectile_end_idx + 1])
    if final_model is None:
        return None
    if boundary_diagnostics is None:
        boundary_diagnostics = _evaluate_projectile_candidate(
            points[accepted_start - 1], points[accepted_start], anchor_points, anchor_model, tolerances,
        )
    release_abs_x = abs(points[accepted_start].x)
    spatial_prior = 1.0 if release_abs_x >= 12.0 else max(0.45, 1.0 - 0.15 * (12.0 - release_abs_x))
    separation = 1.0 - math.exp(-min(5.0, boundary_diagnostics["normalized_error"]))
    confidence = separation * spatial_prior
    if anchor_low_confidence:
        confidence *= 0.75
    return {
        "last_non_projectile_idx": accepted_start - 1,
        "first_projectile_idx": accepted_start,
        "projectile_end_idx": projectile_end_idx,
        "bounce_idx": bounce_idx,
        "projectile_model": final_model,
        "anchor_model": anchor_model,
        "anchor_start_idx": seed_start,
        "por_anchor_point_count": anchor_point_count,
        "por_anchor_low_confidence": anchor_low_confidence,
        "por_x_residual": boundary_diagnostics["x_residual"],
        "por_y_residual": boundary_diagnostics["y_residual"],
        "por_z_residual": boundary_diagnostics["z_residual"],
        "por_xy_line_residual": boundary_diagnostics["xy_line_residual"],
        "por_dx_ratio": boundary_diagnostics["dx_ratio"],
        "por_boundary_normalized_error": boundary_diagnostics["normalized_error"],
        "por_boundary_confidence": confidence,
        "anchor_points": anchor_points,
        "anchor_tolerances": tolerances,
    }


def _interpolated_release(p_before: BallPoint, p_after: BallPoint, alpha: float) -> Dict[str, Any]:
    release_dt = p_before.local_dt + (p_after.local_dt - p_before.local_dt) * alpha
    return {"alpha": float(alpha), "time": release_dt, "point": (
        p_before.x + alpha * (p_after.x - p_before.x),
        p_before.y + alpha * (p_after.y - p_before.y),
        p_before.z + alpha * (p_after.z - p_before.z))}


def estimate_release_midpoint(p_before: BallPoint, p_after: BallPoint) -> Dict[str, Any]:
    return _interpolated_release(p_before, p_after, 0.5)


def estimate_release_solved(
    p_before: BallPoint,
    p_after: BallPoint,
    anchor_model: Optional[ProjectileModel],
) -> Dict[str, Any]:
    """Intersect the extended XY anchor with the boundary perpendicular bisector.

    Release time and height retain the robust midpoint estimate. The horizontal
    point lies on the clean anchor line and is equally distant from the last
    hand-controlled and first projectile samples.
    """
    midpoint = _interpolated_release(p_before, p_after, 0.5)
    if anchor_model is None:
        midpoint.update({"error": None, "fallback": True, "solve_mode": "midpoint_fallback"})
        return midpoint

    a = np.asarray((p_before.x, p_before.y), dtype=float)
    b = np.asarray((p_after.x, p_after.y), dtype=float)
    line_origin = np.asarray((anchor_model.x[0], anchor_model.y[0]), dtype=float)
    line_direction = np.asarray((anchor_model.x[1], anchor_model.y[1]), dtype=float)
    boundary_delta = b - a
    denominator = float(np.dot(line_direction, boundary_delta))
    if abs(denominator) < 1e-9:
        midpoint.update({"error": None, "fallback": True, "solve_mode": "midpoint_fallback"})
        return midpoint

    bisector_rhs = 0.5 * float(np.dot(b, b) - np.dot(a, a))
    line_parameter = (bisector_rhs - float(np.dot(line_origin, boundary_delta))) / denominator
    xy = line_origin + line_parameter * line_direction
    if not np.all(np.isfinite(xy)):
        midpoint.update({"error": None, "fallback": True, "solve_mode": "midpoint_fallback"})
        return midpoint

    point = (float(xy[0]), float(xy[1]), float(midpoint["point"][2]))
    point_array = np.asarray(point)
    before_array = np.asarray((p_before.x, p_before.y, p_before.z))
    after_array = np.asarray((p_after.x, p_after.y, p_after.z))
    equidistance_error = abs(float(np.linalg.norm(point_array - before_array) - np.linalg.norm(point_array - after_array)))
    return {
        "alpha": 0.5,
        "time": midpoint["time"],
        "point": point,
        "error": equidistance_error,
        "fallback": False,
        "solve_mode": "anchor_bisector",
    }


def select_release_index(points: Sequence[BallPoint], start_idx: int, end_idx: int) -> Optional[int]:
    """Compatibility wrapper returning the first backward-detected projectile sample."""
    if not points:
        return None
    backward = detect_backward_projectile_segment(points, start_idx, end_idx)
    return int(backward["first_projectile_idx"]) if backward is not None else None


def compute_velocity_acceleration_from_points(points: Sequence[BallPoint]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Compute per-point speed (m/s) and acceleration magnitude (m/s^2) purely
    from the trajectory's positions using finite differences over time.

    Unlike the fixture CSV's ``speed in m/s`` / ``acceleration in m/s2``
    columns, these values are derived only from the sampled positions (x, y, z)
    and local timestamps of the ball points.

    Speed is the magnitude of the 3D velocity vector: central differences over
    the two-neighbour span for interior points, forward/backward differences at
    the boundaries. Acceleration is the magnitude of the central difference of
    consecutive velocity vectors; it is None for the first and last point where
    no two-sided neighbour support exists.
    """
    n = len(points)
    speeds: List[Optional[float]] = [None] * n
    accels: List[Optional[float]] = [None] * n
    if n < 2:
        return speeds, accels

    # 3D velocity vectors aligned to each point.
    velocities: List[Optional[Tuple[float, float, float]]] = [None] * n

    for i in range(n):
        if i == 0:
            a_idx, b_idx = 0, 1
        elif i == n - 1:
            a_idx, b_idx = n - 2, n - 1
        else:
            a_idx, b_idx = i - 1, i + 1

        dt = (points[b_idx].local_dt - points[a_idx].local_dt).total_seconds()
        if dt <= 0:
            continue
        vx = (points[b_idx].x - points[a_idx].x) / dt
        vy = (points[b_idx].y - points[a_idx].y) / dt
        vz = (points[b_idx].z - points[a_idx].z) / dt
        velocities[i] = (vx, vy, vz)
        speeds[i] = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Acceleration: central difference of the velocity vectors.
    for i in range(1, n - 1):
        v_prev = velocities[i - 1]
        v_next = velocities[i + 1]
        if v_prev is None or v_next is None:
            continue
        dt = (points[i + 1].local_dt - points[i - 1].local_dt).total_seconds()
        if dt <= 0:
            continue
        ax = (v_next[0] - v_prev[0]) / dt
        ay = (v_next[1] - v_prev[1]) / dt
        az = (v_next[2] - v_prev[2]) / dt
        accels[i] = math.sqrt(ax * ax + ay * ay + az * az)

    return speeds, accels


def detect_simple_release_point(
    penalty_row: Dict[str, str],
    positions_dir: Path,
    fixture_cache: Optional[FixtureCache] = None,
    fixture_path: Optional[Path] = None,
    points: Optional[List[BallPoint]] = None,
    normalize_side: bool = False,
    include_deflections: bool = True,
) -> Dict[str, Any]:
    player_id = _coerce_text(penalty_row.get("player_id"))
    goalkeeper_id = _coerce_text(penalty_row.get("goalkeeper_id"))

    # Resolve fixture once if not provided
    if fixture_path is None:
        fixture_index, _ = build_fixture_index(positions_dir)
        fixture_path, _ = resolve_fixture_file(penalty_row, fixture_index, build_edge_case_mappings())
    if fixture_path is None:
        raise FileNotFoundError(f"Could not resolve fixture for {penalty_row.get('id', 'unknown')}")

    raw_start = _coerce_text(penalty_row.get("timestamp_local_timezone"))
    start_dt = parse_penalty_local_time(raw_start)

    # If caller already supplied points, use them directly (avoids reloading).
    if points is None:
        # Prefer cached points when available; otherwise stream from file and only
        # keep points in the time window around the reported start time.
        if fixture_cache is not None:
            try:
                pts = fixture_cache.get_points(fixture_path, start_dt=start_dt, time_window_seconds=5)
            except Exception:
                pts = _build_ball_points_from_file(fixture_path, start_dt=start_dt)
            points = pts
        else:
            points = _build_ball_points_from_file(fixture_path, start_dt=start_dt)
    if start_dt is None:
        start_dt = points[0].local_dt if points else datetime.now()
    if not points:
        raise ValueError(f"No ball points found for penalty {penalty_row.get('id', 'unknown')}")

    # Optionally transform throws performed on the left side of the field (-x)
    # onto the right side (+x) via a 180-degree rotation around the Z axis.
    was_left_side = False
    if normalize_side and _is_left_side(points):
        points = _rotate_180_z(points)
        was_left_side = True

    # Use the reported start time as the anchor and extend the window until the
    # ball clears the goal line, then include a few extra points for context.
    start_idx, end_idx = _find_window(points, start_dt)
    if start_idx >= end_idx:
        start_idx = max(0, end_idx - 2)

    # Optionally skip penalties with a deflection (e.g. by the goalkeeper).
    if not include_deflections and _has_deflection(points, start_idx, end_idx):
        raise ValueError(f"Penalty {penalty_row.get('id', 'unknown')} has a deflection and is excluded")

    segment = detect_backward_projectile_segment(points, start_idx, end_idx)
    if segment is None:
        raise ValueError(f"Penalty {penalty_row.get('id', 'unknown')} has fewer than three usable clean projectile samples")
    first_projectile_idx = int(segment["first_projectile_idx"])
    last_non_projectile_idx = int(segment["last_non_projectile_idx"])
    projectile_end_idx = int(segment["projectile_end_idx"])
    model = segment["projectile_model"]
    release_idx = first_projectile_idx

    release_interval_start_idx = last_non_projectile_idx
    release_interval_end_idx = first_projectile_idx
    before, after = points[release_interval_start_idx], points[release_interval_end_idx]
    midpoint = estimate_release_midpoint(before, after)
    solved = estimate_release_solved(before, after, segment["anchor_model"])
    distance_before_prior = 20.0 - abs(float(solved["point"][0]))
    distance_after_prior = distance_before_prior
    release_boundary_shift = 0
    distance_prior_applied = False

    # The projectile boundary says when free flight is certainly established;
    # the physical release may be one measurement interval earlier. Apply this
    # soft spatial prior without changing any projectile-related index/model.
    if distance_before_prior < 6.4 and last_non_projectile_idx - 1 >= start_idx:
        earlier_start = last_non_projectile_idx - 1
        earlier_end = last_non_projectile_idx
        earlier_before, earlier_after = points[earlier_start], points[earlier_end]
        valid_time = earlier_after.local_dt > earlier_before.local_dt
        if valid_time:
            earlier_solved = estimate_release_solved(earlier_before, earlier_after, segment["anchor_model"])
            earlier_distance = 20.0 - abs(float(earlier_solved["point"][0]))
            earlier_diag = _evaluate_projectile_candidate(
                earlier_before, earlier_after, segment["anchor_points"],
                segment["anchor_model"], segment["anchor_tolerances"],
            )
            tolerances = segment["anchor_tolerances"]
            grossly_implausible = (
                earlier_diag["extreme"]
                or earlier_diag["xy_line_residual"] > 3.0 * tolerances["xy_line"]
            )

            def distance_cost(distance: float) -> float:
                return 0.0 if distance >= 6.4 else ((6.4 - distance) / 0.4) ** 2

            current_cost = float(segment["por_boundary_normalized_error"]) + distance_cost(distance_before_prior)
            earlier_cost = float(earlier_diag["normalized_error"]) + distance_cost(earlier_distance)
            choose_earlier = (
                not grossly_implausible
                and (distance_before_prior < 6.0 or earlier_cost < current_cost)
            )
            if choose_earlier:
                release_interval_start_idx = earlier_start
                release_interval_end_idx = earlier_end
                before, after = earlier_before, earlier_after
                midpoint = estimate_release_midpoint(before, after)
                solved = earlier_solved
                distance_after_prior = earlier_distance
                release_boundary_shift = -1
                distance_prior_applied = True
    release_dt = solved["time"]
    release_xyz = solved["point"]
    fitted_velocity = model.velocity_at(release_dt) if model is not None else (float("nan"),) * 3
    fitted_speed = math.sqrt(sum(v * v for v in fitted_velocity)) if model is not None else None
    fitted_direction = math.degrees(math.atan2(fitted_velocity[1], fitted_velocity[0])) if model is not None else None
    release_point = points[release_idx]
    window_points = points[start_idx:end_idx + 1]
    # Keep both index spaces explicit:
    # - release_idx_global: index in the full loaded points list
    # - release_idx: index relative to trajectory/window_points
    release_idx_global = release_idx
    release_idx_local = max(0, release_idx_global - start_idx)
    trajectory = [
        {
            "t_local": p.local_dt.isoformat(timespec="milliseconds"),
            "ts_ms": p.ts_ms,
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "v": None if math.isnan(p.speed) else p.speed,
            "a": None if math.isnan(p.accel) else p.accel,
            "dir": p.direction,
        }
        for p in window_points
    ]
    # Max velocity/acceleration over the throw trajectory (w.r.t. the ball points).
    finite_vs = [p["v"] for p in trajectory if p["v"] is not None]
    finite_as = [p["a"] for p in trajectory if p["a"] is not None]
    max_v = max(finite_vs) if finite_vs else None
    max_a = max(finite_as) if finite_as else None
    # Velocity/acceleration derived from the positions themselves (finite
    # differences over time), independent of the fixture CSV's speed/accel columns.
    computed_speeds, computed_accels = compute_velocity_acceleration_from_points(window_points)
    return {
        "fixture_file": str(fixture_path),
        "trajectory": trajectory,
        "max_v": max_v,
        "max_a": max_a,
        "velocity_per_point": computed_speeds,
        "acceleration_per_point": computed_accels,
        "release_idx": release_idx_local,
        "release_idx_global": release_idx_global,
        "release_point": {
            "t_local": release_dt.isoformat(timespec="milliseconds"),
            "ts_ms": int(before.ts_ms + solved["alpha"] * (after.ts_ms - before.ts_ms)) if before.ts_ms is not None and after.ts_ms is not None else None,
            "x": release_xyz[0], "y": release_xyz[1], "z": release_xyz[2],
            "v": fitted_speed, "a": None, "dir": fitted_direction,
        },
        "release_speed": fitted_speed,
        "release_accel": None,
        "release_direction": fitted_direction,
        "raw_hbl_release_speed": None if math.isnan(release_point.speed) else release_point.speed,
        "raw_hbl_release_accel": None if math.isnan(release_point.accel) else release_point.accel,
        "raw_hbl_release_direction": release_point.direction,
        "last_non_projectile_idx": last_non_projectile_idx - start_idx,
        "first_projectile_idx": first_projectile_idx - start_idx,
        "projectile_end_idx": projectile_end_idx - start_idx,
        "release_interval_start_idx": release_interval_start_idx - start_idx,
        "release_interval_end_idx": release_interval_end_idx - start_idx,
        "release_boundary_shift": release_boundary_shift,
        "release_distance_prior_applied": distance_prior_applied,
        "release_distance_before_prior": distance_before_prior,
        "release_distance_after_prior": distance_after_prior,
        "por_anchor_start_idx": int(segment["anchor_start_idx"]) - start_idx,
        "por_anchor_point_count": segment["por_anchor_point_count"],
        "por_anchor_low_confidence": segment["por_anchor_low_confidence"],
        "por_x_residual": segment["por_x_residual"],
        "por_y_residual": segment["por_y_residual"],
        "por_z_residual": segment["por_z_residual"],
        "por_xy_line_residual": segment["por_xy_line_residual"],
        "por_dx_ratio": segment["por_dx_ratio"],
        "por_boundary_normalized_error": segment["por_boundary_normalized_error"],
        "por_boundary_confidence": segment["por_boundary_confidence"],
        "release_alpha_midpoint": midpoint["alpha"],
        "release_time_midpoint": midpoint["time"].isoformat(timespec="milliseconds"),
        "release_point_midpoint": {"x": midpoint["point"][0], "y": midpoint["point"][1], "z": midpoint["point"][2]},
        "release_alpha_solved": solved["alpha"],
        "release_solved_error": solved["error"],
        "release_solved_fallback": solved["fallback"],
        "release_solved_mode": solved["solve_mode"],
        "release_time_solved": release_dt.isoformat(timespec="milliseconds"),
        "release_point_solved": {"x": release_xyz[0], "y": release_xyz[1], "z": release_xyz[2]},
        "release_speed_solved": fitted_speed,
        "release_direction_solved": fitted_direction,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "trajectory_points": len(points[start_idx:end_idx + 1]),
        "normalized_side": was_left_side,
    }


def process_penalties_csv(
    penalties_csv: Path,
    positions_dir: Path,
    output_csv: Path,
    errors_csv: Optional[Path] = None,
    include_unsuccessful: bool = False,
    penalty_id: Optional[str] = None,
    normalize_side: bool = False,
    include_deflections: bool = True,
) -> Path:
    with penalties_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    if penalty_id is not None:
        rows = [row for row in rows if _coerce_text(row.get("id")) == penalty_id]

    if not include_unsuccessful:
        rows = [row for row in rows if _coerce_text(row.get("success")) == "1"]

    fixture_cache = FixtureCache()
    records: List[Dict[str, str]] = []
    # Build fixture index once to avoid repeated filesystem scans
    fixture_index, _ = build_fixture_index(positions_dir)
    logger.info("Processing %d penalty rows from %s", len(rows), penalties_csv)
    for row_idx, row in enumerate(rows, start=1):
        try:
            fixture_path, _ = resolve_fixture_file(row, fixture_index, build_edge_case_mappings())
            if fixture_path is None:
                logger.warning(
                    "Row %d (id=%s): could not resolve fixture for %s vs %s",
                    row_idx,
                    _coerce_text(row.get("id")),
                    _coerce_text(row.get("home_team")),
                    _coerce_text(row.get("away_team")),
                )
                result = {"error": f"Could not resolve fixture for {_coerce_text(row.get('id'))}"}
            else:
                logger.info(
                    "Row %d/%d (id=%s): processing fixture %s",
                    row_idx,
                    len(rows),
                    _coerce_text(row.get("id")),
                    fixture_path.name,
                )
                # NOTE: Do NOT preload points once per fixture and reuse them
                # across rows. Each penalty has its own start time, so the ball
                # points must be loaded/filtered for that penalty's own time
                # window. detect_simple_release_point handles this correctly
                # when points=None (it loads via the cache with the row's
                # start_dt). Reusing a single fixture-level window caused the
                # same (first penalty's) points to be applied to every later
                # penalty in the same fixture.
                result = detect_simple_release_point(
                    penalty_row=row,
                    positions_dir=positions_dir,
                    fixture_cache=fixture_cache,
                    fixture_path=fixture_path,
                    normalize_side=normalize_side,
                    include_deflections=include_deflections,
                )
                if result.get("release_idx") is not None:
                    logger.info(
                        "Row %d (id=%s): release detected at idx=%s time=%s",
                        row_idx,
                        _coerce_text(row.get("id")),
                        result.get("release_idx"),
                        result.get("release_point", {}).get("t_local", ""),
                    )
                else:
                    logger.info(
                        "Row %d (id=%s): release detection returned no result",
                        row_idx,
                        _coerce_text(row.get("id")),
                    )
        except Exception as exc:
            logger.error(
                "Row %d (id=%s): error during processing: %s",
                row_idx,
                _coerce_text(row.get("id")),
                exc,
            )
            result = {"error": str(exc)}
        record = {
            "id": _coerce_text(row.get("id")),
            "home_team": _coerce_text(row.get("home_team")),
            "away_team": _coerce_text(row.get("away_team")),
            "player_id": _coerce_text(row.get("player_id")),
            "goalkeeper_id": _coerce_text(row.get("goalkeeper_id")),
            "success": _coerce_text(row.get("success")),
            "distance": _coerce_text(row.get("distance")),
            "timestamp_local_timezone": _coerce_text(row.get("timestamp_local_timezone")),
            "fixture_file": result.get("fixture_file", ""),
            "release_point_json": json.dumps(result.get("release_point", {}), ensure_ascii=False, separators=(",", ":")),
            "release_time_local": result.get("release_point", {}).get("t_local", ""),
            "release_speed": result.get("release_speed"),
            "release_accel": result.get("release_accel"),
            "release_direction": result.get("release_direction"),
            "raw_hbl_release_speed": result.get("raw_hbl_release_speed"),
            "raw_hbl_release_accel": result.get("raw_hbl_release_accel"),
            "raw_hbl_release_direction": result.get("raw_hbl_release_direction"),
            "last_non_projectile_idx": result.get("last_non_projectile_idx"),
            "first_projectile_idx": result.get("first_projectile_idx"),
            "projectile_end_idx": result.get("projectile_end_idx"),
            "release_interval_start_idx": result.get("release_interval_start_idx"),
            "release_interval_end_idx": result.get("release_interval_end_idx"),
            "release_boundary_shift": result.get("release_boundary_shift"),
            "release_distance_prior_applied": result.get("release_distance_prior_applied"),
            "release_distance_before_prior": result.get("release_distance_before_prior"),
            "release_distance_after_prior": result.get("release_distance_after_prior"),
            "por_anchor_start_idx": result.get("por_anchor_start_idx"),
            "por_anchor_point_count": result.get("por_anchor_point_count"),
            "por_anchor_low_confidence": result.get("por_anchor_low_confidence"),
            "por_x_residual": result.get("por_x_residual"),
            "por_y_residual": result.get("por_y_residual"),
            "por_z_residual": result.get("por_z_residual"),
            "por_xy_line_residual": result.get("por_xy_line_residual"),
            "por_dx_ratio": result.get("por_dx_ratio"),
            "por_boundary_normalized_error": result.get("por_boundary_normalized_error"),
            "por_boundary_confidence": result.get("por_boundary_confidence"),
            "release_alpha_midpoint": result.get("release_alpha_midpoint"),
            "release_time_midpoint": result.get("release_time_midpoint"),
            "release_point_midpoint_json": json.dumps(result.get("release_point_midpoint", {}), ensure_ascii=False, separators=(",", ":")),
            "release_alpha_solved": result.get("release_alpha_solved"),
            "release_solved_error": result.get("release_solved_error"),
            "release_solved_fallback": result.get("release_solved_fallback"),
            "release_solved_mode": result.get("release_solved_mode"),
            "release_time_solved": result.get("release_time_solved"),
            "release_point_solved_json": json.dumps(result.get("release_point_solved", {}), ensure_ascii=False, separators=(",", ":")),
            "release_speed_solved": result.get("release_speed_solved"),
            "release_direction_solved": result.get("release_direction_solved"),
            "max_v": result.get("max_v"),
            "max_a": result.get("max_a"),
            "velocity_per_point": result.get("velocity_per_point", []),
            "acceleration_per_point": result.get("acceleration_per_point", []),
            "trajectory_json": json.dumps(result.get("trajectory", []), ensure_ascii=False, separators=(",", ":")),
            "trajectory_point_count": result.get("trajectory_points", 0),
            "release_idx": result.get("release_idx"),
            "release_idx_global": result.get("release_idx_global"),
            "normalized_side": result.get("normalized_side", False),
            "error": result.get("error", ""),
        }
        records.append(record)

    solved_alphas = [
        float(r["release_alpha_solved"])
        for r in records
        if r.get("release_alpha_solved") not in (None, "") and not r.get("release_solved_fallback")
    ]
    if solved_alphas:
        near_one = sum(alpha >= 0.95 for alpha in solved_alphas)
        logger.info(
            "Solved alpha diagnostics: n=%d median=%.3f range=[%.3f, %.3f], alpha>=0.95: %d (%.1f%%)",
            len(solved_alphas), float(np.median(solved_alphas)), min(solved_alphas), max(solved_alphas),
            near_one, 100.0 * near_one / len(solved_alphas),
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "home_team",
        "away_team",
        "player_id",
        "goalkeeper_id",
        "success",
        "distance",
        "timestamp_local_timezone",
        "fixture_file",
        "release_point_json",
        "release_time_local",
        "release_speed",
        "release_accel",
        "release_direction",
        "raw_hbl_release_speed",
        "raw_hbl_release_accel",
        "raw_hbl_release_direction",
        "last_non_projectile_idx",
        "first_projectile_idx",
        "projectile_end_idx",
        "release_interval_start_idx",
        "release_interval_end_idx",
        "release_boundary_shift",
        "release_distance_prior_applied",
        "release_distance_before_prior",
        "release_distance_after_prior",
        "por_anchor_start_idx",
        "por_anchor_point_count",
        "por_anchor_low_confidence",
        "por_x_residual",
        "por_y_residual",
        "por_z_residual",
        "por_xy_line_residual",
        "por_dx_ratio",
        "por_boundary_normalized_error",
        "por_boundary_confidence",
        "release_alpha_midpoint",
        "release_time_midpoint",
        "release_point_midpoint_json",
        "release_alpha_solved",
        "release_solved_error",
        "release_solved_fallback",
        "release_solved_mode",
        "release_time_solved",
        "release_point_solved_json",
        "release_speed_solved",
        "release_direction_solved",
        "max_v",
        "max_a",
        "velocity_per_point",
        "acceleration_per_point",
        "trajectory_json",
        "trajectory_point_count",
        "release_idx",
        "release_idx_global",
        "normalized_side",
        "error",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(records)

    # Separate kinematics CSV in the same output directory: per-throw lists of
    # velocity (m/s) and acceleration (m/s^2) computed from the trajectory's
    # positions (finite differences), one row per penalty id.
    kinematics_csv = output_csv.parent / (output_csv.stem + "_kinematics" + output_csv.suffix)
    with kinematics_csv.open("w", encoding="utf-8", newline="") as handle:
        kin_writer = csv.DictWriter(
            handle,
            fieldnames=["id", "velocity_per_point", "acceleration_per_point"],
            delimiter=";",
        )
        kin_writer.writeheader()
        for record in records:
            kin_writer.writerow(
                {
                    "id": record["id"],
                    "velocity_per_point": json.dumps(
                        record.get("velocity_per_point", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "acceleration_per_point": json.dumps(
                        record.get("acceleration_per_point", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    logger.info("Kinematics CSV written to %s", kinematics_csv)

    errors = [r for r in records if r.get("error")]
    if errors and errors_csv is not None:
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        # write only the error rows for quick inspection
        with errors_csv.open("w", encoding="utf-8", newline="") as eh:
            err_writer = csv.DictWriter(eh, fieldnames=fieldnames, delimiter=";")
            err_writer.writeheader()
            err_writer.writerows(errors)

    logger.info("Processed %d rows -> %d records written to %s (%d with errors)",
                len(rows), len(records), output_csv, len(errors))
    return output_csv


def main() -> None:
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Detect simple point-of-release trajectories for penalties")
    parser.add_argument("--penalties", default=str(project_root / "penalties.csv"))
    parser.add_argument("--positions-dir", default=str(project_root / "games_position_files"))
    parser.add_argument("--output", default=str(project_root / "out" / "simple_penalty_trajectories.csv"))
    parser.add_argument("--include-unsuccessful", action="store_true")
    parser.add_argument("--penalty-id", default=None)
    parser.add_argument(
        "--normalize-side",
        action="store_true",
        help="Transform throws on the left side of the field (-x) onto the right side (+x) via a 180-degree Z rotation",
        default=True
    )
    parser.add_argument(
        "--exclude-deflections",
        action="store_true",
        help="Exclude penalties with a deflection (e.g. by the goalkeeper)",
    )
    args = parser.parse_args()

    # Library callers may configure the release_detector_trajectory_based
    # logger themselves.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create timestamped run directory under out/, matching other pipeline outputs
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "out" / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_csv = run_dir / Path(args.output).name
    errors_csv = run_dir / "simple_penalty_errors.csv"

    process_penalties_csv(
        Path(args.penalties),
        Path(args.positions_dir),
        output_csv,
        errors_csv=errors_csv,
        include_unsuccessful=args.include_unsuccessful,
        penalty_id=args.penalty_id,
        normalize_side=args.normalize_side,
        include_deflections=not args.exclude_deflections,
    )


if __name__ == "__main__":
    main()
