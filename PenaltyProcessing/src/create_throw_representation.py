#!/usr/bin/env python3
"""
Create unified throw representation CSVs for Mocap and League data.

This script processes Mocap throws from one or more throw_type directories and
produces three output CSVs:
  1. raw_mocap.csv   - Mocap throws with PoR-relative trajectories
  2. raw_league.csv  - League throws (loaded from penalties + position files)
  3. throw_index.csv - Throw metadata: id, whole-throw trajectory, PoR index,
                       global segment/PoR indices

Directory layout (per throw type):
  mocap_files/<throw_type>/
    ├── 6DOF/       # ball marker files (*6DOF_3D.tsv)
    ├── skeleton/   # skeleton files (*s_Josh.tsv)
    └── body/       # body marker files (*_labeling_done.tsv, optional)

Each throw_type directory is one recording that may contain multiple throws,
detected via mocap_por_detection_pipeline.detect_throw_segments().

Coordinate transformation:
  1. Swap X/Y (newX = oldY, newY = -oldX). Mocap PoR/free-flight detection
     (mocap_por_detection_pipeline) assumes the ball trajectory already uses
     this swapped, +X-is-throw-direction axis convention (e.g. its wall-hit
     detection checks for an X-velocity sign reversal), so
     process_mocap_recording always resolves (or creates on demand) the
     genuinely swapped *_XY_swapped.tsv files and treats the loaded ball
     centres as already swapped. --use-swap-xy only controls whether an
     existing *_XY_swapped.tsv is reused instead of being regenerated; it no
     longer changes which coordinate transform is applied.
  2. Translate: Mocap (0,0,0) → League (12.6, 0, 0) i.e. +12.6m in x
     after the swap, so the 7m line (League x=13) aligns with Mocap origin.
  3. Convert mm → m (Mocap data is in millimeters)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse the X/Y swap utility from the tools directory (needed to guarantee
# genuinely swapped Mocap files, see process_mocap_recording).
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

logger = logging.getLogger("create_throw_representation")


def setup_logging(output_dir: Path, verbose: bool) -> Path:
    """Configure console + file logging; returns the log file path.

    Attaches the handlers to the root logger (rather than only the
    "create_throw_representation" logger) so log calls from dependency
    modules invoked during the pipeline - "penalty_processing" and
    "release_detector_trajectory_based" (both default to WARNING/no handlers
    when used as a library, per their own module docstrings) - propagate up
    and are actually printed/written instead of silently disappearing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "create_throw_representation.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # Raise the level on the dependency loggers explicitly - they default to
    # WARNING and won't emit INFO/DEBUG messages otherwise, even though they
    # now propagate to the root logger's handlers above.
    for dependency_logger_name in ("penalty_processing", "release_detector_trajectory_based"):
        logging.getLogger(dependency_logger_name).setLevel(level)

    logger.setLevel(level)
    return log_path

from ball_trajectory import (
    BallPoint,
    load_ball_points,
    load_goalkeeper_candidates,
    load_possession_context_candidates,
    first_idx_at_or_after,
    first_idx_at_or_before,
    find_goalline_crossing_idx,
    detect_physics_based_release_point,
    is_plausible_release_point,
    adjust_start_idx_by_distance,
    serialize_trajectory,
    serialize_point,
)
from fixture_resolution import build_fixture_index, resolve_fixture_file, build_edge_case_mappings
from penalty_time_utils import parse_penalty_local_time, parse_position_local_time, try_float, try_int
from release_detector_trajectory_based import (
    compute_velocity_acceleration_from_points,
    find_bounce_idx,
    _rotate_180_z,
    _is_left_side,
)


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

# League canonical court: origin at court centre, +x toward goal
# Goal line at x = +20 m, 7 m line at x = +13 m
# Mocap origin (0,0,0) maps to (13 m - 0.40 m, 0, 0) in League coordinates.
# → after swap_xy, translate Mocap x by +12.6 m

MOCAP_TO_LEAGUE_TRANSLATION_M = 12.6  # mm→m handled below; pure x-translation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThrowRepresentation:
    """Unified representation of a throw (Mocap or League)."""
    throw_id: str
    source: str  # "mocap" or "league"
    # Canonical (League) coordinates in metres
    por_position: Tuple[float, float, float]  # (x, y, z) in m
    release_speed_m_s: float
    release_direction_deg: float
    release_height_m: float
    # Trajectory: PoR-relative, starting at (0,0,0)
    trajectory_json: str  # JSON of BallPoint list (PoR-relative, in m), each
    # point carries a "frame" key = frame index relative to PoR (frame 0 = PoR)
    # at the source's native sampling rate, see sampling_rate_hz.
    trajectory_point_count: int
    trajectory_duration_ms: float
    # Native sampling rate of the source (300.0 for Mocap, 20.0 for League),
    # needed to downsample Mocap frames to the League rate later.
    sampling_rate_hz: float = 0.0
    is_bounce: Optional[bool] = None
    bounce_index_in_trajectory: Optional[int] = None
    release_timing_type: str = "native"
    first_projectile_offset_ms: Optional[float] = None
    # Full-throw info (for throw_index.csv)
    full_trajectory_json: Optional[str] = None
    full_trajectory_point_count: Optional[int] = None
    por_index_in_full: Optional[int] = None
    segment_start_global_idx: Optional[int] = None
    por_global_idx: Optional[int] = None
    # Error/validity
    valid: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def mocap_to_league_coords(
    x_mm: float, y_mm: float, z_mm: float, already_swapped: bool = False
) -> Tuple[float, float, float]:
    """
    Convert Mocap coordinates (mm) to League canonical coordinates (m).

    Step 1: Swap X/Y (newX = oldY, newY = -oldX) — applied here unless
             ``already_swapped=True`` (i.e. the input file is a *_XY_swapped.tsv
             variant, in which case the swap is skipped to avoid double-swapping).
    Step 2: Convert mm → m.
    Step 3: Translate x by +12.6 m so Mocap (0,0,0) → League (12.6, 0, 0)
             (at the calibrated seven-metre-line position).

    Returns (x_League_m, y_League_m, z_League_m).
    """
    if already_swapped:
        # File is already swapped: no swap step, just mm→m + translate
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        z_m = z_mm / 1000.0
    else:
        # Step 1: swap (newX = oldY, newY = -oldX), still in mm
        x_swapped_mm = y_mm
        y_swapped_mm = -x_mm

        # Step 2: convert mm → m (translation below is expressed in metres)
        x_m = x_swapped_mm / 1000.0
        y_m = y_swapped_mm / 1000.0
        z_m = z_mm / 1000.0

    # Step 3: translate x by +12.6 m (Mocap origin → 7m line in League)
    x_m += MOCAP_TO_LEAGUE_TRANSLATION_M

    return x_m, y_m, z_m


def league_to_mocap_coords(
    x_m: float, y_m: float, z_m: float, already_swapped: bool = False
) -> Tuple[float, float, float]:
    """
    Reverse transformation: League canonical → Mocap original (mm).
    Useful for debugging / validation.
    """
    # Reverse: convert m → mm, translate -12.6m, then un-swap
    x_mm = (x_m - MOCAP_TO_LEAGUE_TRANSLATION_M) * 1000.0
    y_mm = y_m * 1000.0
    z_mm = z_m * 1000.0
    if already_swapped:
        return x_mm, y_mm, z_mm
    # un-swap: originalX = -newY, originalY = newX
    orig_x = -y_mm
    orig_y = x_mm
    orig_z = z_mm
    return orig_x, orig_y, orig_z


# ---------------------------------------------------------------------------
# Trajectory utilities
# ---------------------------------------------------------------------------

def compute_kinematics_from_points(
    points: List[BallPoint],
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Compute per-point velocity (m/s), acceleration (m/s²), horizontal direction
    (deg), velocity components (m/s), and vertical launch angle (deg) from
    BallPoint positions using the SAME methodology as
    compute_velocity_acceleration_from_points in release_detector_trajectory_based.py.

    Vertical launch angle is theta_elevation = atan2(vz, sqrt(vx^2 + vy^2)),
    i.e. the angle of the velocity vector above (+) or below (-) the horizontal
    (x-y) plane, in degrees.

    Returns (speeds, accels, directions, vxs, vys, vzs, vert_angles_deg) aligned
    to input points. Values may be None where computation is not possible.
    """
    n = len(points)
    if n < 2:
        return [None] * n, [None] * n, [None] * n, [None] * n, [None] * n, [None] * n, [None] * n

    # 3D velocity vectors via central differences
    velocities: List[Optional[Tuple[float, float, float]]] = [None] * n
    speeds: List[Optional[float]] = [None] * n
    directions: List[Optional[float]] = [None] * n
    acceleration_magnitudes: List[Optional[float]] = [None] * n
    vxs: List[Optional[float]] = [None] * n
    vys: List[Optional[float]] = [None] * n
    vzs: List[Optional[float]] = [None] * n
    vert_angles: List[Optional[float]] = [None] * n

    # Forward/backward differences at boundaries, central otherwise
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
        directions[i] = math.degrees(math.atan2(vy, vx))
        vxs[i] = vx
        vys[i] = vy
        vzs[i] = vz
        vert_angles[i] = math.degrees(math.atan2(vz, math.sqrt(vx * vx + vy * vy)))

    # Acceleration: central difference of velocity vectors
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
        acceleration_magnitudes[i] = math.sqrt(ax * ax + ay * ay + az * az)

    # Boundary acceleration: forward/backward difference
    if n >= 2:
        # First point: difference between point 1 and point 0 velocities
        if velocities[0] is not None and velocities[1] is not None:
            dt = (points[1].local_dt - points[0].local_dt).total_seconds()
            if dt > 0:
                ax = (velocities[1][0] - velocities[0][0]) / dt
                ay = (velocities[1][1] - velocities[0][1]) / dt
                az = (velocities[1][2] - velocities[0][2]) / dt
                acceleration_magnitudes[0] = math.sqrt(ax * ax + ay * ay + az * az)

        # Last point: difference between point n-1 and point n-2 velocities
        if velocities[n - 2] is not None and velocities[n - 1] is not None:
            dt = (points[n - 1].local_dt - points[n - 2].local_dt).total_seconds()
            if dt > 0:
                ax = (velocities[n - 1][0] - velocities[n - 2][0]) / dt
                ay = (velocities[n - 1][1] - velocities[n - 2][1]) / dt
                az = (velocities[n - 1][2] - velocities[n - 2][2]) / dt
                acceleration_magnitudes[n - 1] = math.sqrt(ax * ax + ay * ay + az * az)

    return speeds, acceleration_magnitudes, directions, vxs, vys, vzs, vert_angles


def po_relative_trajectory(
    points: List[BallPoint],
    por_idx: int,
    frame_numbers: Optional[List[float]] = None,
) -> Tuple[List[BallPoint], List[float], List[float], List[float], List[float], List[float]]:
    """
    Return PoR-relative trajectory starting from (0,0,0).

    For each point at index i: p_rel = points[i] - points[por_idx]
    Also returns t_since_release, speed, acceleration, direction for each point.

    The returned points have x_rel, y_rel, z_rel in metres (input points are
    already in metres/canonical coords at this stage, no further unit
    conversion is applied here).

    Args:
        points: Input trajectory points (already in metres).
        por_idx: Index of the Point of Release within ``points``.
        frame_numbers: Optional native sample index per point (e.g. the
            actual Mocap tsv frame number, which may have gaps). Defaults to
            ``range(len(points))``, appropriate for already-uniformly-sampled
            data (e.g. League positions).

    Returns:
        (rel_points, t_since_ms, speeds, accels, dirs, frame_rel) where
        ``frame_rel[i] = frame_numbers[i] - frame_numbers[por_idx]``, so
        ``frame_rel[por_idx] == 0`` (frame 0 = PoR).
    """
    if por_idx is None or por_idx < 0 or por_idx >= len(points):
        return [], [], [], [], [], []

    if frame_numbers is None:
        frame_numbers = list(range(len(points)))

    por_point = points[por_idx]
    por_frame = frame_numbers[por_idx]

    rel_points: List[BallPoint] = []
    t_since: List[float] = []
    speeds: List[Optional[float]] = []
    accels: List[Optional[float]] = []
    dirs: List[Optional[float]] = []
    frame_rel: List[float] = []

    for i, p in enumerate(points):
        # PoR-relative position (points already in metres)
        x_rel = p.x - por_point.x
        y_rel = p.y - por_point.y
        z_rel = p.z - por_point.z

        # Time since PoR
        dt = (p.local_dt - por_point.local_dt).total_seconds()
        t_s = dt * 1000.0  # ms

        rel_points.append(
            BallPoint(
                local_dt=p.local_dt,
                ts_ms=p.ts_ms,
                x=x_rel,
                y=y_rel,
                z=z_rel,
                speed=p.speed,
                accel=p.accel,
                direction=p.direction,
            )
        )
        t_since.append(t_s)
        speeds.append(p.speed)
        accels.append(p.accel)
        dirs.append(p.direction)
        frame_rel.append(frame_numbers[i] - por_frame)

    return rel_points, t_since, speeds, accels, dirs, frame_rel


def _clean_float(value: Optional[float]) -> Optional[float]:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def serialize_trajectory_with_frames(
    points: List[BallPoint],
    frame_rel: List[float],
    t_since_ms: Optional[List[float]] = None,
    vxs: Optional[List[Optional[float]]] = None,
    vys: Optional[List[Optional[float]]] = None,
    vzs: Optional[List[Optional[float]]] = None,
    vert_angles: Optional[List[Optional[float]]] = None,
) -> str:
    """Serialize PoR-relative points, embedding the PoR-relative frame index.

    ``frame_rel[i]`` is the sample index relative to the PoR (frame 0 = PoR)
    at the source's native sampling rate (see ``ThrowRepresentation.sampling_rate_hz``).
    ``vxs``/``vys``/``vzs`` are per-point velocity components (m/s) and
    ``vert_angles`` is the vertical launch angle (deg,
    atan2(vz, sqrt(vx^2+vy^2))); all default to None per point if omitted.
    """
    payload = []
    for i, (p, fr) in enumerate(zip(points, frame_rel)):
        speed = p.speed
        accel = p.accel
        payload.append(
            {
                "frame": fr,
                "t_since_ms": t_since_ms[i] if t_since_ms is not None and i < len(t_since_ms) else None,
                "t_local": p.local_dt.isoformat(timespec="milliseconds"),
                "ts_ms": p.ts_ms,
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "v": None if speed is None or (isinstance(speed, float) and math.isnan(speed)) else speed,
                "a": None if accel is None or (isinstance(accel, float) and math.isnan(accel)) else accel,
                "dir": p.direction,
                "vx": _clean_float(vxs[i]) if vxs is not None and i < len(vxs) else None,
                "vy": _clean_float(vys[i]) if vys is not None and i < len(vys) else None,
                "vz": _clean_float(vzs[i]) if vzs is not None and i < len(vzs) else None,
                "vert_angle": _clean_float(vert_angles[i]) if vert_angles is not None and i < len(vert_angles) else None,
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def find_goal_line_crossing_idx(points: List[BallPoint], start_idx: int = 0) -> Optional[int]:
    """
    Return the first index where |x| > 20 m (goal line crossed).
    Goal lines are at x = +20 and x = -20 in League canonical coords.
    """
    for i in range(max(0, start_idx), len(points)):
        p = points[i]
        if abs(p.x) > 20.0:
            return i
    return None


# ---------------------------------------------------------------------------
# League throw loading
# ---------------------------------------------------------------------------

LEAGUE_SAMPLING_RATE_HZ = 20.0
MOCAP_SAMPLING_RATE_HZ = 300.0


def _angular_abs_diff_deg(a: float, b: float) -> float:
    """Absolute difference between two angles in degrees, wrapped to [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def compare_direction_recomputation(
    throw_id: str,
    original_dirs: List[Optional[float]],
    recomputed_dirs: List[Optional[float]],
) -> Dict[str, Any]:
    """
    Compare the source-provided direction values against directions recomputed
    with ``compute_kinematics_from_points`` (atan2(vy, vx) in canonical coords).

    Returns a summary dict with mean/max absolute angular difference in degrees,
    so the caller can decide (via --direction-source) whether to keep the
    original values or switch to the recomputed ones.
    """
    diffs = []
    for o, r in zip(original_dirs, recomputed_dirs):
        if o is None or r is None:
            continue
        diffs.append(_angular_abs_diff_deg(o, r))

    if not diffs:
        return {
            "throw_id": throw_id,
            "n_compared": 0,
            "mean_abs_diff_deg": None,
            "max_abs_diff_deg": None,
        }

    return {
        "throw_id": throw_id,
        "n_compared": len(diffs),
        "mean_abs_diff_deg": sum(diffs) / len(diffs),
        "max_abs_diff_deg": max(diffs),
    }


def load_league_throws(
    penalties_file: Path,
    positions_dir: Path,
    limit: Optional[int] = None,
    include_unsuccessful: bool = False,
    penalty_id: Optional[str] = None,
    direction_source: str = "original",
    precomputed_csv: Optional[Path] = None,
) -> Tuple[List[ThrowRepresentation], List[Dict[str, Any]]]:
    """
    Load League throws from penalties.csv and position files using the
    trajectory-based release detector, returning ThrowRepresentation objects
    with PoR-relative trajectories.

    Args:
        direction_source: unused - retained only for backward-compatible CLI
            usage. Every kinematic feature (speed/accel/direction) is now
            always taken from ``compute_kinematics_from_points`` (recomputed
            from the trajectory positions), never the source-provided values,
            for both League and Mocap. A diagnostic comparison between the
            recomputed and source-provided direction is still returned so the
            difference can be inspected.
        precomputed_csv: Optional path to a previously computed
            simple_penalty_trajectories.csv (from release_detector_trajectory_based.py).
            If provided, the expensive per-penalty PoR detection is skipped and
            the throws are loaded directly from this file.

    Returns:
        (representations, direction_diff_records) - the second element has one
        entry per throw summarizing how much the recomputed direction differs
        from the original.
    """
    if precomputed_csv is not None:
        # Skip the expensive pipeline; load from the precomputed CSV directly.
        if not precomputed_csv.exists():
            logger.warning("Precomputed League CSV %s does not exist", precomputed_csv)
            return [], []
        output_file = precomputed_csv
    else:
        from penalty_processing import process_penalties

        # Run the pipeline to get the basic output
        run_dir = process_penalties(
            penalties_file=penalties_file,
            positions_dir=positions_dir,
            output_dir=Path("./out_induced_by_pipeline"),
            tol_y=0.35,
            tol_z=0.35,
            limit=limit,
            include_unsuccessful=include_unsuccessful,
            penalty_id=penalty_id,
        )

        # Read the output CSV
        output_file = run_dir / "penalty_trajectories.csv"
        if not output_file.exists():
            return [], []

    df = pd.read_csv(output_file, sep=";", dtype=str)
    representations: List[ThrowRepresentation] = []
    direction_diff_records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        throw_id = row.get("id", "")
        if not throw_id:
            continue

        try:
            # Parse release point
            release_point_json = row.get("release_point_json", "{}")
            release_point = json.loads(release_point_json)

            por_x = try_float(release_point.get("x", "0"))
            por_y = try_float(release_point.get("y", "0"))
            por_z = try_float(release_point.get("z", "0"))
            por_dt = parse_position_local_time(release_point.get("t_local", ""))

            # Parse trajectory
            traj_json = row.get("trajectory_json", "[]")
            trajectory = json.loads(traj_json)

            # Convert trajectory points to BallPoint-like structure
            # and apply coordinate transformation + PoR-relative
            ball_points: List[BallPoint] = []
            for pt in trajectory:
                x = try_float(pt.get("x", "0"))
                y = try_float(pt.get("y", "0"))
                z = try_float(pt.get("z", "0"))
                # These are already in League canonical coords from the pipeline
                ball_points.append(
                    BallPoint(
                        local_dt=parse_position_local_time(pt.get("t_local", "")),
                        ts_ms=try_int(pt.get("ts_ms", "")),
                        x=x,
                        y=y,
                        z=z,
                        speed=try_float(pt.get("v", "")),
                        accel=try_float(pt.get("a", "")),
                        direction=try_float(pt.get("dir", "")),
                    )
                )

            if not ball_points:
                continue

            # Normalize side: if the throw is on the left side of the field
            # (-x), rotate 180° around Z so all throws are on the +x side.
            # This is needed because the precomputed simple_penalty_trajectories.csv
            # may have been generated without --normalize-side.
            if _is_left_side(ball_points):
                ball_points = _rotate_180_z(ball_points)
                por_x, por_y, por_z = -por_x, -por_y, por_z

            # The detected PoR is synthetic and normally lies halfway between
            # two measured samples, so it must be inserted as a real trajectory
            # point rather than replaced by ``release_idx`` (the first measured
            # projectile sample).
            first_projectile_idx = try_int(row.get("first_projectile_idx", ""))
            if first_projectile_idx is None:
                first_projectile_idx = try_int(row.get("release_idx", ""))
            if first_projectile_idx is None or not 0 <= first_projectile_idx < len(ball_points):
                raise ValueError("League CSV has no valid first_projectile_idx/release_idx")
            # ``projectile_end_idx`` belongs to PoR detection and deliberately
            # stops before a detected bounce. Do not reuse it for reconstruction: the
            # raw League continuation must retain the bounce and rebound until
            # the first goal-line crossing. This leaves release detection
            # completely unchanged while giving reconstruction the full path.
            continuation_end_idx = find_goal_line_crossing_idx(ball_points, first_projectile_idx)
            if continuation_end_idx is None:
                continuation_end_idx = len(ball_points) - 1
            if por_dt is None:
                before_idx = max(0, first_projectile_idx - 1)
                por_dt = ball_points[before_idx].local_dt + (
                    ball_points[first_projectile_idx].local_dt - ball_points[before_idx].local_dt
                ) * 0.5

            synthetic_por = BallPoint(
                local_dt=por_dt,
                ts_ms=try_int(release_point.get("ts_ms", "")),
                x=float(por_x), y=float(por_y), z=float(por_z),
                speed=try_float(row.get("release_speed_solved", row.get("release_speed", ""))) or float("nan"),
                accel=float("nan"),
                direction=try_float(row.get("release_direction_solved", row.get("release_direction", ""))),
            )
            ball_points = [synthetic_por, *ball_points[first_projectile_idx:continuation_end_idx + 1]]
            por_idx = 0
            bounce_idx = find_bounce_idx(ball_points, por_idx, len(ball_points) - 1)
            # Fractional League-frame coordinates preserve the half-frame first
            # interval: normally 0, 0.5, 1.5, 2.5, ... at 20 Hz.
            league_frames = [
                (point.local_dt - por_dt).total_seconds() * LEAGUE_SAMPLING_RATE_HZ
                for point in ball_points
            ]
            rel_points, t_since, rel_speeds, rel_accels, rel_dirs, frame_rel = po_relative_trajectory(
                ball_points, por_idx, frame_numbers=league_frames
            )

            # trajectory duration (from first to last point)
            if len(t_since) >= 2:
                duration_ms = t_since[-1] - t_since[0]
            else:
                duration_ms = 0.0

            # Unify kinematic recomputation with Mocap: recompute speed/accel/dir
            # from the PoR-relative positions using the same finite-difference
            # methodology, instead of trusting the source-provided values.
            (
                computed_speeds, computed_accels, computed_dirs,
                computed_vxs, computed_vys, computed_vzs, computed_vert_angles,
            ) = compute_kinematics_from_points(rel_points)

            # Diagnostic: how much does the recomputed direction differ from
            # the direction already present in the League source data?
            diff_stats = compare_direction_recomputation(throw_id, rel_dirs, computed_dirs)
            direction_diff_records.append(diff_stats)

            # Every kinematic feature (speed/accel/direction) must come from
            # the recomputed trajectory-based kinematics, never the
            # source-provided values, for both League and Mocap.
            final_speeds, final_accels, final_dirs = computed_speeds, computed_accels, computed_dirs

            # Release info (at PoR, index por_idx in rel_points)
            release_speed = final_speeds[por_idx] if por_idx < len(final_speeds) and final_speeds[por_idx] is not None else 0.0
            release_dir = final_dirs[por_idx] if por_idx < len(final_dirs) and final_dirs[por_idx] is not None else 0.0
            release_height = por_z

            # Embed the unified kinematics + PoR-relative frame index into the
            # serialized points before writing them out.
            enhanced_rel_points: List[BallPoint] = []
            for i, p in enumerate(rel_points):
                enhanced_rel_points.append(
                    BallPoint(
                        local_dt=p.local_dt,
                        ts_ms=p.ts_ms,
                        x=p.x,
                        y=p.y,
                        z=p.z,
                        speed=final_speeds[i] if i < len(final_speeds) else None,
                        accel=final_accels[i] if i < len(final_accels) else None,
                        direction=final_dirs[i] if i < len(final_dirs) else None,
                    )
                )

            # Serialize PoR-relative trajectory (with frame index + velocity
            # components/vertical angle, always recomputed regardless of
            # --direction-source since they aren't provided by the source data)
            traj_ser = serialize_trajectory_with_frames(
                enhanced_rel_points, frame_rel, t_since_ms=t_since,
                vxs=computed_vxs, vys=computed_vys, vzs=computed_vzs, vert_angles=computed_vert_angles,
            ) if enhanced_rel_points else "[]"

            rep = ThrowRepresentation(
                throw_id=throw_id,
                source="league",
                por_position=(por_x, por_y, por_z),
                release_speed_m_s=float(release_speed) if release_speed is not None else 0.0,
                release_direction_deg=float(release_dir) if release_dir is not None else 0.0,
                release_height_m=float(release_height) if release_height is not None else 0.0,
                trajectory_json=traj_ser,
                trajectory_point_count=len(rel_points),
                trajectory_duration_ms=float(duration_ms),
                sampling_rate_hz=LEAGUE_SAMPLING_RATE_HZ,
                is_bounce=bounce_idx is not None,
                bounce_index_in_trajectory=bounce_idx,
                release_timing_type=(
                    "half_frame_adjacent" if len(t_since) > 1 and 15.0 <= t_since[1] <= 35.0
                    else "half_frame_shifted" if len(t_since) > 1 and 65.0 <= t_since[1] <= 85.0
                    else "on_frame" if len(t_since) > 1 and 40.0 <= t_since[1] <= 60.0
                    else "irregular"
                ),
                first_projectile_offset_ms=float(t_since[1]) if len(t_since) > 1 else None,
                full_trajectory_json=traj_json,
                full_trajectory_point_count=len(trajectory),
                por_index_in_full=None,  # synthetic PoR is not present in the source full trajectory
                segment_start_global_idx=None,  # League has no "segment start" in same sense
                por_global_idx=None,
                valid=True,
            )
            representations.append(rep)

        except Exception as exc:
            logger.exception("Error processing league throw %s: %s", throw_id, exc)
            continue

    return representations, direction_diff_records


def load_league_throws_from_csv(csv_path: Path) -> List[ThrowRepresentation]:
    """
    Load League ThrowRepresentation objects from a previously written
    raw_league.csv (or any CSV with the same schema). This avoids re-running
    the expensive League pipeline when processing multiple throw types.
    """
    if not csv_path.exists():
        logger.warning("League CSV %s does not exist", csv_path)
        return []

    df = pd.read_csv(csv_path, sep=";", dtype=str)
    representations: List[ThrowRepresentation] = []

    for _, row in df.iterrows():
        throw_id = row.get("throw_id", "")
        if not throw_id:
            continue

        try:
            por_x = try_float(row.get("por_x_m", "0"))
            por_y = try_float(row.get("por_y_m", "0"))
            por_z = try_float(row.get("por_z_m", "0"))

            rep = ThrowRepresentation(
                throw_id=throw_id,
                source="league",
                por_position=(por_x, por_y, por_z),
                release_speed_m_s=try_float(row.get("release_speed_m_s", "0")) or 0.0,
                release_direction_deg=try_float(row.get("release_direction_deg", "0")) or 0.0,
                release_height_m=try_float(row.get("release_height_m", "0")) or 0.0,
                trajectory_json=row.get("trajectory_json", "[]"),
                trajectory_point_count=try_int(row.get("trajectory_point_count", "0")) or 0,
                trajectory_duration_ms=try_float(row.get("trajectory_duration_ms", "0")) or 0.0,
                sampling_rate_hz=try_float(row.get("sampling_rate_hz", "20.0")) or LEAGUE_SAMPLING_RATE_HZ,
                is_bounce=_parse_optional_bool(row.get("is_bounce")),
                bounce_index_in_trajectory=try_int(row.get("bounce_index_in_trajectory", "")),
                release_timing_type=row.get("release_timing_type", "native") or "native",
                first_projectile_offset_ms=try_float(row.get("first_projectile_offset_ms", "")),
                valid=(row.get("valid", "True").strip().lower() == "true"),
                error=row.get("error", ""),
            )
            representations.append(rep)
        except Exception as exc:
            logger.exception("Error loading league throw %s from CSV: %s", throw_id, exc)
            continue

    logger.info("Loaded %d League throws from %s", len(representations), csv_path)
    return representations


def _parse_optional_bool(value: Any) -> Optional[bool]:
    """Parse CSV booleans while preserving a genuinely missing value."""
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


# ---------------------------------------------------------------------------
# Mocap throw processing
# ---------------------------------------------------------------------------

def _find_file(dir_path: Path, pattern: str, use_swapped: bool) -> Optional[Path]:
    """
    Find a file in dir_path matching pattern, optionally preferring the
    XY_swapped variant.

    If use_swapped=True, prefer files with '_XY_swapped' in the name.
    If use_swapped=False (default), prefer files WITHOUT '_XY_swapped'.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return None

    matches = sorted(dir_path.glob(pattern))
    if not matches:
        return None

    if use_swapped:
        swapped = [p for p in matches if "XY_swapped" in p.name]
        return swapped[0] if swapped else matches[0]
    else:
        non_swapped = [p for p in matches if "XY_swapped" not in p.name]
        return non_swapped[0] if non_swapped else matches[0]


def _find_or_create_swapped_file(dir_path: Path, pattern: str) -> Optional[Path]:
    """
    Find the '_XY_swapped' variant of a file matching pattern, creating it via
    swap_xy_file() if no swapped variant exists yet.

    mocap_por_detection_pipeline's free-flight/wall-hit detection assumes the
    loaded ball trajectory already uses the swapped (+X = throw direction)
    axis convention. Previously, _find_file(use_swapped=True) silently fell
    back to a non-swapped file if no '_XY_swapped' variant existed, which
    left the trajectory in the raw axis convention: wall-hit detection then
    checked the wrong (lateral) axis for a velocity-sign reversal, and the
    release direction ended up rotated ~90° from the expected value. Always
    resolving to a genuinely swapped file removes that failure mode.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return None

    matches = sorted(dir_path.glob(pattern))
    if not matches:
        return None

    swapped = [p for p in matches if "XY_swapped" in p.name]
    if swapped:
        return swapped[0]

    raw = [p for p in matches if "XY_swapped" not in p.name]
    if not raw:
        return None

    from swap_xy_tsv import swap_xy_file  # type: ignore

    return swap_xy_file(raw[0])


def process_mocap_recording(
    recording_dir: Path,
    throw_type: str,
    use_swapped: bool = False,
) -> List[ThrowRepresentation]:
    """
    Process a single Mocap recording directory and return a list of
    ThrowRepresentation objects — one per detected throw segment.

    The recording_dir should contain:
      - 6DOF/       containing *6DOF_3D.tsv (ball markers → sphere fit)
      - skeleton/   containing *s_Josh.tsv (skeleton)
      - body/       containing *_labeling_done.tsv (optional)

    The ball and skeleton files are always resolved to their genuinely
    swapped '_XY_swapped.tsv' variant (creating it on demand if missing),
    because mocap_por_detection_pipeline's PoR/free-flight detection assumes
    the +X axis is the throw direction. ``use_swapped`` no longer changes the
    coordinate transform; it is unused for the ball/skeleton files (kept for
    backward-compatible signature / body-file lookup only).

    Returns a list of ThrowRepresentation (one per detected throw segment).
    """
    # Find the relevant files in their subdirectories
    sixdof_dir = recording_dir / "6DOF"
    skeleton_dir = recording_dir / "skeleton"
    body_dir = recording_dir / "body"

    ball_path = _find_or_create_swapped_file(sixdof_dir, "*6DOF_3D.tsv")
    skeleton_path = _find_or_create_swapped_file(skeleton_dir, "*s_Josh.tsv")
    body_path = _find_file(body_dir, "*_labeling_done.tsv", use_swapped)

    if ball_path is None or skeleton_path is None:
        logger.warning(
            "Missing files in %s (ball=%s, skeleton=%s), skipping",
            recording_dir, ball_path, skeleton_path,
        )
        return []

    logger.debug(
        "Recording %s: ball=%s skeleton=%s body=%s",
        throw_type, ball_path.name, skeleton_path.name,
        body_path.name if body_path else "None",
    )

    # Index the files
    from mocap_por_detection_pipeline import (
        index_skeleton_file, index_ball_3d_file, index_ball_6d_file,
    )
    from mocap_por_detection_pipeline import parse_skeleton_row_cached

    ball_index = index_ball_3d_file(ball_path)
    skeleton_index = index_skeleton_file(skeleton_path)

    # Find common frames
    shared_frames = sorted(set(skeleton_index.frames) & set(ball_index.frames))
    if not shared_frames:
        logger.warning("No shared frames in %s, skipping", recording_dir)
        return []

    # Load ball samples per frame (sphere fit from 6DOF marker data)
    # and skeleton positions per frame (needed for hand-ball distance).
    from mocap_por_detection_pipeline import (
        _load_ball_sample_from_fit, _load_ball_sample_from_gt,
    )

    ball_by_frame: Dict[int, Any] = {}
    positions_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    segment_names = tuple(skeleton_index.segment_names or ())

    for frame in shared_frames:
        skeleton_offset = skeleton_index.frame_to_offset.get(frame)
        if skeleton_offset is None:
            continue
        _, time_s, positions = parse_skeleton_row_cached(
            str(skeleton_index.path), skeleton_offset, segment_names
        )
        positions_by_frame[frame] = positions

        # Pass the real time_s so BallPoint timestamps are correct; otherwise
        # all points share the same timestamp and kinematics (v/a/dir) become
        # None because dt = 0.
        ball_by_frame[frame] = _load_ball_sample_from_fit(frame, time_s, ball_index)

    # Detect throw segments
    from mocap_por_detection_pipeline import detect_throw_segments

    segments = detect_throw_segments(
        frames=shared_frames,
        positions_by_frame=positions_by_frame,
        ball_by_frame=ball_by_frame,
        reacquire_distance_mm=120.0,
        sigma_multiplier=6.0,
        baseline_min_samples=100,
        baseline_std_floor_mm=1.0,
    )

    if not segments:
        logger.warning("No throw segments detected in %s, skipping", recording_dir)
        return []

    logger.info("Detected %d throw segments in %s", len(segments), throw_type)

    representations: List[ThrowRepresentation] = []

    for seg_idx, segment in enumerate(segments):
        throw_id = f"{throw_type}_seg{seg_idx + 1}"

        # Collect the full window samples from segment start to end.
        # Extend the window to capture post-release flight data, matching the
        # original run_pipeline behaviour: collect up to the start of the next
        # segment (or the end of all frames) so the trajectory includes the
        # ball's flight after release.
        from mocap_por_detection_pipeline import _collect_window_samples

        next_start = segments[seg_idx + 1].start_frame if seg_idx + 1 < len(segments) else shared_frames[-1] + 1
        segment_search_end = max(segment.end_frame, next_start - 1)

        valid_window_samples, _ = _collect_window_samples(
            ball_by_frame=ball_by_frame,
            all_frames=shared_frames,
            start_frame=segment.start_frame,
            end_frame=segment_search_end,
        )

        if not valid_window_samples:
            logger.warning("No valid window samples for %s, skipping", throw_id)
            continue

        # Detect PoR using method 1 (the 6-sigma hand-ball distance spike)
        from mocap_por_detection_pipeline import _compute_kinematics

        full_kinematics = _compute_kinematics(valid_window_samples, fs_hz=300.0)
        method1 = _method_result_at_frame(full_kinematics, segment.method1_por_frame)

        por_frame = method1.por_frame if method1.por_frame is not None else segment.method1_por_frame

        # Truncate to the free-flight window (PoR -> wall hit / ball becomes
        # unidentified), exactly as detected by mocap_por_detection_pipeline,
        # so post-release trajectories don't run into the next throw's window.
        from mocap_por_detection_pipeline import _detect_free_flight_window

        _, free_flight_end = _detect_free_flight_window(
            ball_by_frame=ball_by_frame,
            all_frames=shared_frames,
            por_frame=por_frame,
            samples=valid_window_samples,
            kinematics=full_kinematics,
        )
        if free_flight_end is not None:
            # Clamp defensively so the PoR sample itself is never truncated
            # away (the -5 frame buffer in _detect_free_flight_window can, in
            # rare near-instant-impact cases, land just before por_frame).
            free_flight_end = max(free_flight_end, por_frame)
            valid_window_samples = [s for s in valid_window_samples if s.frame <= free_flight_end]

        # Find the index of the PoR in our samples
        por_idx_in_samples = None
        for i, sample in enumerate(valid_window_samples):
            if sample.frame == por_frame:
                por_idx_in_samples = i
                break

        if por_idx_in_samples is None:
            # Fallback: use the index from the segment
            por_idx_in_samples = 0

        # Get the PoR point
        por_sample = valid_window_samples[por_idx_in_samples] if por_idx_in_samples < len(valid_window_samples) else valid_window_samples[0]
        por_x_m, por_y_m, por_z_m = mocap_to_league_coords(
            por_sample.center_mm[0], por_sample.center_mm[1], por_sample.center_mm[2],
            already_swapped=True,
        )

        # Build BallPoint list from samples, applying the same canonical
        # (swap + translate + mm->m) coordinate transform as the PoR point, and
        # tracking the native (possibly gapped) Mocap frame number per point.
        all_ball_points: List[BallPoint] = []
        all_ball_frames: List[int] = []
        for i, sample in enumerate(valid_window_samples):
            if sample.center_mm is None:
                continue
            x_m, y_m, z_m = mocap_to_league_coords(
                sample.center_mm[0], sample.center_mm[1], sample.center_mm[2],
                already_swapped=True,
            )
            origin = datetime(1970, 1, 1)
            all_ball_points.append(
                BallPoint(
                    local_dt=origin + timedelta(seconds=sample.time_s),
                    ts_ms=sample.frame,  # approximate
                    x=x_m,
                    y=y_m,
                    z=z_m,
                    speed=float("nan"),
                    accel=float("nan"),
                    direction=None,
                )
            )
            all_ball_frames.append(sample.frame)

        if not all_ball_points:
            logger.warning("No valid ball points for %s, skipping", throw_id)
            continue

        # Locate the PoR within all_ball_points/all_ball_frames by frame number
        # (not by list position in valid_window_samples, since points with a
        # missing center_mm are skipped above and would otherwise shift indices).
        por_idx_in_points = None
        for i, fr in enumerate(all_ball_frames):
            if fr == por_frame:
                por_idx_in_points = i
                break
        if por_idx_in_points is None:
            por_idx_in_points = 0

        # PoR-relative trajectory; frame_numbers carries the true (possibly
        # gapped) 300 Hz Mocap frame index so frame_rel reflects real elapsed
        # samples at MOCAP_SAMPLING_RATE_HZ, with frame 0 = PoR.
        rel_points, t_since, rel_speeds, rel_accels, rel_dirs, frame_rel = po_relative_trajectory(
            all_ball_points, por_idx_in_points, frame_numbers=all_ball_frames
        )

        # Trajectory duration
        if len(t_since) >= 2:
            duration_ms = t_since[-1] - t_since[0]
        else:
            duration_ms = 0.0

        # Compute kinematics from the PoR-relative trajectory using the unified method
        (
            computed_speeds, computed_accels, computed_dirs,
            computed_vxs, computed_vys, computed_vzs, computed_vert_angles,
        ) = compute_kinematics_from_points(rel_points)

        # Release speed at PoR
        release_speed = computed_speeds[por_idx_in_points] if por_idx_in_points < len(computed_speeds) and computed_speeds[por_idx_in_points] is not None else 0.0

        # Release direction at PoR
        release_dir = computed_dirs[por_idx_in_points] if por_idx_in_points < len(computed_dirs) and computed_dirs[por_idx_in_points] is not None else 0.0

        # Serialize PoR-relative trajectory with kinematics
        enhanced_rel_points: List[BallPoint] = []
        for i, p in enumerate(rel_points):
            enhanced_rel_points.append(
                BallPoint(
                    local_dt=p.local_dt,
                    ts_ms=p.ts_ms,
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    speed=computed_speeds[i] if computed_speeds and i < len(computed_speeds) else None,
                    accel=computed_accels[i] if computed_accels and i < len(computed_accels) else None,
                    direction=computed_dirs[i] if computed_dirs and i < len(computed_dirs) else None,
                )
            )

        traj_ser = serialize_trajectory_with_frames(
            enhanced_rel_points, frame_rel, t_since_ms=t_since,
            vxs=computed_vxs, vys=computed_vys, vzs=computed_vzs, vert_angles=computed_vert_angles,
        ) if enhanced_rel_points else "[]"

        # Full trajectory (whole throw, not PoR-relative)
        full_traj_ser = serialize_trajectory(all_ball_points) if all_ball_points else "[]"
        bounce_idx = find_bounce_idx(
            all_ball_points, por_idx_in_points, len(all_ball_points) - 1,
        )

        # Determine global indices
        # segment_start_global_idx: index in the full shared_frames list
        segment_start_global_idx = None
        for i, frame in enumerate(shared_frames):
            if frame == segment.start_frame:
                segment_start_global_idx = i
                break

        # por_global_idx: global index of PoR
        por_global_idx = None
        for i, frame in enumerate(shared_frames):
            if frame == por_frame:
                por_global_idx = i
                break

        # Release height
        release_height_m = por_z_m

        rep = ThrowRepresentation(
            throw_id=throw_id,
            source="mocap",
            por_position=(por_x_m, por_y_m, por_z_m),
            release_speed_m_s=float(release_speed) if release_speed is not None else 0.0,
            release_direction_deg=float(release_dir) if release_dir is not None else 0.0,
            release_height_m=float(release_height_m) if release_height_m is not None else 0.0,
            trajectory_json=traj_ser,
            trajectory_point_count=len(rel_points),
            trajectory_duration_ms=float(duration_ms),
            sampling_rate_hz=MOCAP_SAMPLING_RATE_HZ,
            is_bounce=bounce_idx is not None,
            bounce_index_in_trajectory=bounce_idx,
            release_timing_type="dense_native",
            first_projectile_offset_ms=(t_since[por_idx_in_points + 1] if por_idx_in_points + 1 < len(t_since) else None),
            full_trajectory_json=full_traj_ser,
            full_trajectory_point_count=len(all_ball_points),
            por_index_in_full=por_idx_in_points,
            segment_start_global_idx=segment_start_global_idx,
            por_global_idx=por_global_idx,
            valid=True,
        )

        representations.append(rep)
        logger.debug(
            "  -> %s: por at (%.3f, %.3f, %.3f) m, %d points",
            throw_id,
            rep.por_position[0], rep.por_position[1], rep.por_position[2],
            rep.trajectory_point_count,
        )

    return representations


def _method_result_at_frame(kinematics, frame):
    """Helper: get MethodResult at a given frame (copied from mocap_por_detection_pipeline)."""
    if frame is None:
        return None
    try:
        idx = kinematics.frames.index(frame)
    except ValueError:
        return None
    return type('MethodResult', (), {
        'por_frame': kinematics.frames[idx],
        'velocity_m_s': kinematics.speed_mm_s[idx] / 1000.0 if kinematics.speed_mm_s[idx] is not None else None,
        'acceleration_m_s2': kinematics.acceleration_mm_s2[idx] / 1000.0 if kinematics.acceleration_mm_s2[idx] is not None else None,
        'direction_deg': kinematics.direction_deg[idx],
    })()


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_raw_mocap_csv(representations: List[ThrowRepresentation], output_path: Path) -> None:
    """Write raw_mocap.csv with PoR-relative trajectory data."""
    fieldnames = [
        "throw_id",
        "source",
        "por_x_m", "por_y_m", "por_z_m",
        "release_speed_m_s", "release_direction_deg", "release_height_m",
        "trajectory_json",
        "trajectory_point_count", "trajectory_duration_ms", "sampling_rate_hz",
        "is_bounce", "bounce_index_in_trajectory",
        "release_timing_type", "first_projectile_offset_ms",
        "valid", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rep in representations:
            if rep.source != "mocap":
                continue
            row = {
                "throw_id": rep.throw_id,
                "source": rep.source,
                "por_x_m": rep.por_position[0],
                "por_y_m": rep.por_position[1],
                "por_z_m": rep.por_position[2],
                "release_speed_m_s": rep.release_speed_m_s,
                "release_direction_deg": rep.release_direction_deg,
                "release_height_m": rep.release_height_m,
                "trajectory_json": rep.trajectory_json,
                "trajectory_point_count": rep.trajectory_point_count,
                "trajectory_duration_ms": rep.trajectory_duration_ms,
                "sampling_rate_hz": rep.sampling_rate_hz,
                "is_bounce": "" if rep.is_bounce is None else int(rep.is_bounce),
                "bounce_index_in_trajectory": rep.bounce_index_in_trajectory,
                "release_timing_type": rep.release_timing_type,
                "first_projectile_offset_ms": rep.first_projectile_offset_ms,
                "valid": rep.valid,
                "error": rep.error,
            }
            writer.writerow(row)


def write_raw_league_csv(representations: List[ThrowRepresentation], output_path: Path) -> None:
    """Write raw_league.csv with PoR-relative trajectory data."""
    fieldnames = [
        "throw_id",
        "source",
        "por_x_m", "por_y_m", "por_z_m",
        "release_speed_m_s", "release_direction_deg", "release_height_m",
        "trajectory_json",
        "trajectory_point_count", "trajectory_duration_ms", "sampling_rate_hz",
        "is_bounce", "bounce_index_in_trajectory",
        "release_timing_type", "first_projectile_offset_ms",
        "valid", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rep in representations:
            if rep.source != "league":
                continue
            row = {
                "throw_id": rep.throw_id,
                "source": rep.source,
                "por_x_m": rep.por_position[0],
                "por_y_m": rep.por_position[1],
                "por_z_m": rep.por_position[2],
                "release_speed_m_s": rep.release_speed_m_s,
                "release_direction_deg": rep.release_direction_deg,
                "release_height_m": rep.release_height_m,
                "trajectory_json": rep.trajectory_json,
                "trajectory_point_count": rep.trajectory_point_count,
                "trajectory_duration_ms": rep.trajectory_duration_ms,
                "sampling_rate_hz": rep.sampling_rate_hz,
                "is_bounce": "" if rep.is_bounce is None else int(rep.is_bounce),
                "bounce_index_in_trajectory": rep.bounce_index_in_trajectory,
                "release_timing_type": rep.release_timing_type,
                "first_projectile_offset_ms": rep.first_projectile_offset_ms,
                "valid": rep.valid,
                "error": rep.error,
            }
            writer.writerow(row)


def write_throw_index_csv(representations: List[ThrowRepresentation], output_path: Path) -> None:
    """
    Write throw_index.csv with throw metadata:
    - throw_id
    - trajectory_json_list of the WHOLE mocap throw (from segment start to end)
    - index relative to that list where the PoR is
    - global indices for segment start, PoR
    - Also includes league data with appropriate fields
    """
    fieldnames = [
        "throw_id",
        "source",
        "trajectory_json_list",  # whole-throw trajectory JSON
        "por_index_relative",    # index within trajectory_json_list where PoR is
        "segment_start_global_idx",
        "por_global_idx",
        "por_x_m", "por_y_m", "por_z_m",
        "release_speed_m_s",
        "release_direction_deg",
        "release_height_m",
        "is_bounce",
        "bounce_index_in_trajectory",
        "trajectory_point_count_full",
        "trajectory_duration_ms_full",
        "valid", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for rep in representations:
            row = {
                "throw_id": rep.throw_id,
                "source": rep.source,
                "trajectory_json_list": rep.full_trajectory_json or "[]",
                "por_index_relative": rep.por_index_in_full,
                "segment_start_global_idx": rep.segment_start_global_idx,
                "por_global_idx": rep.por_global_idx,
                "por_x_m": rep.por_position[0],
                "por_y_m": rep.por_position[1],
                "por_z_m": rep.por_position[2],
                "release_speed_m_s": rep.release_speed_m_s,
                "release_direction_deg": rep.release_direction_deg,
                "release_height_m": rep.release_height_m,
                "is_bounce": "" if rep.is_bounce is None else int(rep.is_bounce),
                "bounce_index_in_trajectory": rep.bounce_index_in_trajectory,
                "trajectory_point_count_full": rep.full_trajectory_point_count or 0,
                "trajectory_duration_ms_full": rep.trajectory_duration_ms,
                "valid": rep.valid,
                "error": rep.error,
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create unified throw representation CSVs for Mocap and League data. "
            "Processes Mocap throws from one or more throw_type directories and "
            "produces: raw_mocap.csv, raw_league.csv, throw_index.csv"
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory of the PenaltyProcessing project",
    )
    parser.add_argument(
        "--throw-type",
        type=str,
        nargs="+",
        default=["throw_ul"],
        help="One or more throw type subdirectories under root-dir/mocap_files/ "
             "(e.g., 'throw_ul throw_wurfvar5'). Each is a recording that may "
             "contain multiple throws.",
    )
    parser.add_argument(
        "--use-swap-xy",
        action="store_true",
        default=False,
        help="Deprecated / no longer affects correctness: Mocap ball and "
             "skeleton files are always resolved to their genuinely swapped "
             "*_XY_swapped.tsv variant (created on demand if missing), since "
             "mocap_por_detection_pipeline's wall-hit/free-flight detection "
             "requires the +X-is-throw-direction axis convention. Kept only "
             "for backward-compatible CLI usage.",
    )
    parser.add_argument(
        "--league-penalties",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "penalties.csv",
        help="Path to penalties.csv for League data",
    )
    parser.add_argument(
        "--league-positions-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "games_position_files",
        help="Directory containing *_2_phases_positions.csv for League data",
    )
    parser.add_argument(
        "--league-csv",
        type=Path,
        default=None,
        help="Optional path to a previously computed simple_penalty_trajectories.csv "
             "(from release_detector_trajectory_based.py). If provided, the expensive "
             "per-penalty PoR detection is skipped and League throws are loaded "
             "directly from this file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of penalties to process",
    )
    parser.add_argument(
        "--include-unsuccessful",
        action="store_true",
        help="Include unsuccessful throws in League processing",
    )
    parser.add_argument(
        "--penalty-id",
        type=str,
        default=None,
        help="If set, only process the penalty row with this id",
    )
    parser.add_argument(
        "--direction-source",
        choices=["original", "recomputed"],
        default="original",
        help=(
            "Deprecated / no longer affects correctness: every kinematic "
            "feature (speed/accel/direction) for League throws is always "
            "recomputed from the trajectory positions with the same "
            "finite-difference method used for Mocap. A diagnostic comparison "
            "against the source-provided direction is always written/printed. "
            "Kept only for backward-compatible CLI usage."
        ),
    )
    parser.add_argument(
        "--extract-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run feature extraction on the raw CSVs at the end of the pipeline (default: on)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "throw_features",
        help="Base output directory for the CSV files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information",
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir

    log_path = setup_logging(output_dir, args.verbose)

    logger.info("=== Create Throw Representation ===")
    logger.info("Root dir: %s", root_dir)
    logger.info("Throw types: %s", ", ".join(args.throw_type))
    logger.info("Use swapped XY files: %s", args.use_swap_xy)
    logger.info("Output dir: %s", output_dir)
    logger.info("Log file: %s", log_path)

    # Process Mocap throws.
    logger.info("--- Processing Mocap throws ---")

    mocap_reps: List[ThrowRepresentation] = []
    n_mocap_failed = 0

    for throw_type in args.throw_type:
        recording_dir = root_dir / "mocap_files" / throw_type
        if not recording_dir.exists() or not recording_dir.is_dir():
            logger.warning("Throw type directory %s does not exist, skipping", recording_dir)
            n_mocap_failed += 1
            continue

        logger.info("Processing recording %s...", throw_type)
        try:
            reps = process_mocap_recording(
                recording_dir=recording_dir,
                throw_type=throw_type,
                use_swapped=args.use_swap_xy,
            )
        except Exception:
            logger.exception("Unhandled error processing Mocap recording %s (%s)", throw_type, recording_dir)
            n_mocap_failed += 1
            continue

        if reps:
            mocap_reps.extend(reps)
        else:
            n_mocap_failed += 1
            logger.debug("  -> No throws detected in %s", throw_type)

    logger.info("Successfully processed %d Mocap throws across %d recording(s) (%d failed/skipped)",
                len(mocap_reps), len(args.throw_type), n_mocap_failed)

    # Load League throws.
    logger.info("--- Loading League throws ---")

    # Compute League throws once per invocation. If --league-csv is provided,
    # skip the expensive per-penalty PoR detection and load from the
    # precomputed simple_penalty_trajectories.csv instead.
    league_reps, direction_diff_records = load_league_throws(
        penalties_file=args.league_penalties,
        positions_dir=args.league_positions_dir,
        limit=args.limit,
        include_unsuccessful=args.include_unsuccessful,
        penalty_id=args.penalty_id,
        direction_source=args.direction_source,
        precomputed_csv=args.league_csv,
    )
    logger.info("Successfully loaded %d League throws", len(league_reps))

    # Compare source-provided and recomputed League directions.
    compared = [r for r in direction_diff_records if r["n_compared"] > 0]
    if compared:
        mean_diffs = [r["mean_abs_diff_deg"] for r in compared]
        max_diffs = [r["max_abs_diff_deg"] for r in compared]
        logger.info("--- Direction recomputation check (original vs. recomputed) ---")
        logger.info("Throws compared: %d / %d", len(compared), len(direction_diff_records))
        logger.info("Mean of per-throw mean |diff|: %.2f deg", sum(mean_diffs) / len(mean_diffs))
        logger.info("Max of per-throw max |diff|:   %.2f deg", max(max_diffs))
        logger.info("All kinematic features use recomputed trajectory-based kinematics (not the source-provided values)")

    direction_diff_path = output_dir / "direction_recomputation_check.csv"
    with direction_diff_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["throw_id", "n_compared", "mean_abs_diff_deg", "max_abs_diff_deg"], delimiter=";"
        )
        writer.writeheader()
        writer.writerows(direction_diff_records)
    logger.info("Wrote direction recomputation diagnostics to %s", direction_diff_path)

    # Write raw trajectory outputs.
    logger.info("--- Writing output CSVs ---")

    raw_mocap_path = output_dir / "raw_mocap.csv"
    write_raw_mocap_csv(mocap_reps, raw_mocap_path)
    logger.info("Wrote %d Mocap rows to %s", len(mocap_reps), raw_mocap_path)

    raw_league_path = output_dir / "raw_league.csv"
    write_raw_league_csv(league_reps, raw_league_path)
    logger.info("Wrote %d League rows to %s", len(league_reps), raw_league_path)

    throw_index_path = output_dir / "throw_index.csv"
    write_throw_index_csv(mocap_reps, throw_index_path)
    logger.info("Wrote %d rows to %s", len(mocap_reps), throw_index_path)

    # Extract matching features unless explicitly disabled.
    features_mocap_path: Optional[Path] = None
    features_league_path: Optional[Path] = None
    if args.extract_features:
        logger.info("--- Extracting matching features ---")
        from feature_extraction import extract_features_csv

        features_mocap_path = output_dir / "features_mocap.csv"
        n_mocap_features = extract_features_csv(raw_mocap_path, features_mocap_path)
        logger.info("Wrote %d rows to %s", n_mocap_features, features_mocap_path)

        features_league_path = output_dir / "features_league.csv"
        n_league_features = extract_features_csv(raw_league_path, features_league_path)
        logger.info("Wrote %d rows to %s", n_league_features, features_league_path)

    # ---- Summary ----
    logger.info("=== Summary ===")
    logger.info("Mocap throws processed: %d", len(mocap_reps))
    logger.info("League throws loaded: %d", len(league_reps))
    logger.info("Output files:")
    logger.info("  %s", raw_mocap_path)
    logger.info("  %s", raw_league_path)
    logger.info("  %s", throw_index_path)
    logger.info("  %s", direction_diff_path)
    if features_mocap_path is not None:
        logger.info("  %s", features_mocap_path)
        logger.info("  %s", features_league_path)
    logger.info("=== Done ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
