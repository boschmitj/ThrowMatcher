"""
Ball trajectory loading and release-point detection.

This module is the core of the penalty analysis pipeline. It is responsible for:

- Loading ball, goalkeeper, and possession-context points from a single
  ``*_2_phases_positions.csv`` file (one fixture).
- Selecting the goalkeeper trajectory that best matches a given ball trajectory.
- Detecting the "release point" (point of release, PoR) of a penalty throw,
  i.e. the moment the ball leaves the thrower's hand. This is done either with
  a simple heuristic (``select_release_point``) or with a physics-based
  projectile fit (``detect_physics_based_release_point``).
- Serializing trajectories to JSON for downstream visualization.

Coordinate system note: the handball court is 40 m long (x from -20 to +20)
and 20 m wide (y from -10 to +10). The goal lines are at x = -20 and x = +20.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from penalty_time_utils import parse_position_local_time, try_float, try_int


# Module-level logger. Defaults to WARNING so library use does not spam stdout
# unless the caller opts in (e.g. via logging.basicConfig(level=logging.INFO)).
logger = logging.getLogger("ball_trajectory")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BallPoint:
    """A single tracked position of the ball.

    Attributes:
        local_dt: Local wall-clock timestamp of the sample.
        ts_ms: Raw millisecond timestamp from the tracking system (may be None).
        x, y, z: Position in meters on the court (x: -20..20, y: -10..10).
        speed: Reported speed in m/s (NaN if the source did not provide it).
        accel: Reported acceleration in m/s^2 (NaN if not provided).
        direction: Reported direction of movement in degrees (None if missing).
    """
    local_dt: datetime
    ts_ms: Optional[int]
    x: float
    y: float
    z: float
    speed: float
    accel: float
    direction: Optional[float]


@dataclass
class GoalkeeperPoint:
    """A single tracked position of a goalkeeper candidate.

    Attributes:
        local_dt: Local wall-clock timestamp of the sample.
        ts_ms: Raw millisecond timestamp (may be None).
        x, y, z: Position in meters on the court.
        sensor_id: Identifier of the sensor/tracking tag this point belongs to.
    """
    local_dt: datetime
    ts_ms: Optional[int]
    x: float
    y: float
    z: float
    sensor_id: str


@dataclass
class PossessionContextPoint:
    """A non-ball point that carries ball-possession information.

    Used as contextual evidence for the release point: the player who is about
    to throw should be the one holding the ball (possession) right before the
    release, and should no longer hold it right after.

    Attributes:
        local_dt: Local wall-clock timestamp of the sample.
        ts_ms: Raw millisecond timestamp (may be None).
        x, y, z: Position in meters on the court.
        speed: Reported speed in m/s (NaN if not provided).
        accel: Reported acceleration in m/s^2 (NaN if not provided).
        direction: Reported direction of movement in degrees (None if missing).
        group_name: Tracking group the point belongs to (e.g. a team name).
        full_name: Human-readable name of the tracked person.
        possession_id: Id of the ball this person currently possesses.
    """
    local_dt: datetime
    ts_ms: Optional[int]
    x: float
    y: float
    z: float
    speed: float
    accel: float
    direction: Optional[float]
    group_name: str
    full_name: str
    sensor_id: str
    possession_id: str


@dataclass
class ReleaseDetectionResult:
    """Result of the physics-based release point detection.

    Attributes:
        release_idx: Index into the input trajectory of the detected release
            point, or None if no release point could be found.
        candidate_window: (start, end) index range that was searched.
        goal_cutoff_idx: First index where the ball is within
            ``goal_distance_limit_m`` of the goal line (None if never reached).
        ground_contact_idx: First index where the ball appears to touch the
            ground (None if not detected).
        release_point: The detected release BallPoint (None if not found).
        release_velocity: Dict with vx/vy/vz/speed of the fitted projectile at
            release (None if not found).
        release_distance_to_goal: Distance in meters from the release point to
            the goal line (None if not found).
        release_score: Composite score of the best candidate.
        projectile_score: Sub-score from how well the segment fits a projectile.
        kinematic_score: Sub-score from velocity/gravity consistency.
        core_score: Sub-score from possession-context evidence.
        distance_score: Sub-score rewarding a release near the 7 m line.
        confidence: Final 0..1 confidence in the detected release point.
        diagnostics: Extra info for debugging (candidate scores, rank gap, ...).
    """
    release_idx: Optional[int]
    candidate_window: Tuple[int, int]
    goal_cutoff_idx: Optional[int]
    ground_contact_idx: Optional[int]
    release_point: Optional[BallPoint]
    release_velocity: Optional[Dict[str, float]]
    release_distance_to_goal: Optional[float]
    release_score: float
    projectile_score: float
    kinematic_score: float
    core_score: float
    distance_score: float
    confidence: float
    diagnostics: Dict[str, Any]


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_ball_points(positions_file: Path) -> List[BallPoint]:
    """Load only Ball rows with required columns from one positions file.

    Reads a ``*_2_phases_positions.csv`` file, keeps only the rows whose
    "group name" is exactly "Ball", parses the numeric columns, and returns the
    points sorted by time. Rows with a missing/invalid time or position are
    skipped.

    Args:
        positions_file: Path to a single fixture positions CSV.

    Returns:
        Chronologically sorted list of BallPoint objects.

    Raises:
        ValueError: If a required column is missing from the CSV header.
    """
    points: List[BallPoint] = []

    with positions_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        # Columns that must exist in the file for this loader to work.
        required = {
            "formatted local time",
            "group name",
            "x in m",
            "y in m",
            "z in m",
            "speed in m/s",
            "acceleration in m/s2",
            "direction of movement in deg",
            "ts in ms",
        }
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {positions_file.name}: {missing}")

        for row in reader:
            # Only keep rows that actually describe the ball.
            if (row.get("group name") or "").strip() != "Ball":
                continue

            # Parse each field; try_* helpers return None for empty/invalid values.
            local_dt = parse_position_local_time(row.get("formatted local time", ""))
            x = try_float(row.get("x in m", ""))
            y = try_float(row.get("y in m", ""))
            z = try_float(row.get("z in m", ""))
            speed = try_float(row.get("speed in m/s", ""))
            accel = try_float(row.get("acceleration in m/s2", ""))
            direction = try_float(row.get("direction of movement in deg", ""))
            ts_ms = try_int(row.get("ts in ms", ""))

            # A point without a valid time or position is unusable -> skip it.
            if local_dt is None or x is None or y is None or z is None:
                continue

            points.append(
                BallPoint(
                    local_dt=local_dt,
                    ts_ms=ts_ms,
                    x=x,
                    y=y,
                    z=z,
                    # Missing speed/accel are stored as NaN so downstream
                    # math (e.g. max()) does not break.
                    speed=speed if speed is not None else float("nan"),
                    accel=accel if accel is not None else float("nan"),
                    direction=direction,
                )
            )

    # Sort by time so downstream index-based logic is valid.
    points.sort(key=lambda p: (p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


def load_goalkeeper_candidates(positions_file: Path) -> List[GoalkeeperPoint]:
    """Load potential goalkeeper points from one positions file.

    We keep only points close to either goal area so downstream matching remains fast.

    A goalkeeper is expected to stand near one of the two goals. This loader
    therefore keeps only non-ball points whose x is within 16..20 m of either
    goal line and whose y is within 1.5 m of the center line. This drastically
    reduces the number of candidate points that later have to be matched.

    Args:
        positions_file: Path to a single fixture positions CSV.

    Returns:
        List of GoalkeeperPoint candidates, sorted by (sensor_id, time).
    """
    points: List[GoalkeeperPoint] = []

    with positions_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        required = {
            "formatted local time",
            "group name",
            "x in m",
            "y in m",
            "z in m",
            "ts in ms",
        }
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {positions_file.name}: {missing}")

        for row in reader:
            # Skip the ball itself; we only want player/sensor points.
            if (row.get("group name") or "").strip() == "Ball":
                continue

            local_dt = parse_position_local_time(row.get("formatted local time", ""))
            x = try_float(row.get("x in m", ""))
            y = try_float(row.get("y in m", ""))
            z = try_float(row.get("z in m", ""))
            ts_ms = try_int(row.get("ts in ms", ""))

            if local_dt is None or x is None or y is None or z is None:
                continue

            # Restrict to likely goalkeeper area near either goal.
            # abs(x) in [16, 20] -> close to a goal line; abs(y) <= 1.5 -> near center.
            if not (16.0 <= abs(x) <= 20.0 and abs(y) <= 1.5):
                continue

            # Prefer the most specific identifier available in the row.
            sensor_id = (
                (row.get("sensor id") or "").strip()
                or (row.get("mapped id") or "").strip()
                or (row.get("full name") or "").strip()
                or "unknown"
            )

            points.append(
                GoalkeeperPoint(
                    local_dt=local_dt,
                    ts_ms=ts_ms,
                    x=x,
                    y=y,
                    z=z,
                    sensor_id=sensor_id,
                )
            )

    # Sort per sensor by time so per-sensor alignment logic is valid.
    points.sort(key=lambda p: (p.sensor_id, p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


def load_possession_context_candidates(positions_file: Path) -> List[PossessionContextPoint]:
    """Load non-ball rows for contextual thrower evidence.

    The possession field is kept even when empty so the detector can observe
    when a player transitions from possessing the ball to not possessing it.
    This is diagnostic context only; it does not decide release by itself.

    Args:
        positions_file: Path to a single fixture positions CSV.

    Returns:
        Chronologically sorted list of PossessionContextPoint objects.
    """
    points: List[PossessionContextPoint] = []

    with positions_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        required = {
            "formatted local time",
            "group name",
            "x in m",
            "y in m",
            "z in m",
            "speed in m/s",
            "acceleration in m/s2",
            "direction of movement in deg",
            "ts in ms",
            "ball possession (id of possessed ball)",
        }
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in {positions_file.name}: {missing}")

        for row in reader:
            # Skip the ball itself.
            if (row.get("group name") or "").strip() == "Ball":
                continue

            possession_id = (row.get("ball possession (id of possessed ball)") or "").strip()

            local_dt = parse_position_local_time(row.get("formatted local time", ""))
            x = try_float(row.get("x in m", ""))
            y = try_float(row.get("y in m", ""))
            z = try_float(row.get("z in m", ""))
            speed = try_float(row.get("speed in m/s", ""))
            accel = try_float(row.get("acceleration in m/s2", ""))
            direction = try_float(row.get("direction of movement in deg", ""))
            ts_ms = try_int(row.get("ts in ms", ""))

            if local_dt is None or x is None or y is None or z is None:
                continue

            points.append(
                PossessionContextPoint(
                    local_dt=local_dt,
                    ts_ms=ts_ms,
                    x=x,
                    y=y,
                    z=z,
                    speed=speed if speed is not None else float("nan"),
                    accel=accel if accel is not None else float("nan"),
                    direction=direction,
                    group_name=(row.get("group name") or "").strip(),
                    full_name=(row.get("full name") or "").strip(),
                    sensor_id=(row.get("sensor id") or "").strip(),
                    possession_id=possession_id,
                )
            )

    points.sort(key=lambda p: (p.local_dt, p.ts_ms if p.ts_ms is not None else -1))
    return points


# ---------------------------------------------------------------------------
# Goalkeeper trajectory selection
# ---------------------------------------------------------------------------

def _detect_shot_direction(points: List[BallPoint]) -> int:
    """Return the side of the court the shot is heading towards.

    Returns +1 if the ball is (on average) on the positive x side (goal at
    x=+20), or -1 if it is on the negative x side (goal at x=-20). This is used
    to pick the goalkeeper on the correct side of the field.
    """
    if not points:
        return 1
    avg_x = sum(p.x for p in points) / len(points)
    return -1 if avg_x < 0 else 1


def _align_goalkeeper_to_ball_timestamps(
    sensor_points: List[GoalkeeperPoint],
    ball_traj: List[BallPoint],
    sensor_id: str,
) -> List[GoalkeeperPoint]:
    """Build goalkeeper samples exactly at ball timestamps (same window, same count).

    The goalkeeper tracking and the ball tracking may not share the exact same
    sample times. This function resamples the goalkeeper trajectory onto the
    ball's timestamp grid so that both trajectories have the same length and
    can be compared/plotted point-by-point. For each ball timestamp it picks
    the nearest goalkeeper sample (by time).

    Args:
        sensor_points: Goalkeeper points of a single sensor, sorted by time.
        ball_traj: The ball trajectory whose timestamps define the output grid.
        sensor_id: Identifier to stamp onto the output points.

    Returns:
        A list of GoalkeeperPoint with the same length as ``ball_traj``, each
        carrying the ball's timestamp but the goalkeeper's position.
    """
    if not sensor_points or not ball_traj:
        return []

    aligned: List[GoalkeeperPoint] = []
    idx = 0

    # Nearest-neighbour alignment also handles gaps in goalkeeper tracking.
    for ball_point in ball_traj:
        # Advance the goalkeeper pointer until it passes the current ball time.
        while idx + 1 < len(sensor_points) and sensor_points[idx + 1].local_dt <= ball_point.local_dt:
            idx += 1

        # Choose the closer of the two surrounding goalkeeper samples.
        best = sensor_points[idx]
        if idx + 1 < len(sensor_points):
            left_delta = abs((ball_point.local_dt - sensor_points[idx].local_dt).total_seconds())
            right_delta = abs((sensor_points[idx + 1].local_dt - ball_point.local_dt).total_seconds())
            if right_delta < left_delta:
                best = sensor_points[idx + 1]

        # Output point uses the ball's timestamp but the goalkeeper's position.
        aligned.append(
            GoalkeeperPoint(
                local_dt=ball_point.local_dt,
                ts_ms=ball_point.ts_ms,
                x=best.x,
                y=best.y,
                z=best.z,
                sensor_id=sensor_id,
            )
        )

    return aligned


def extract_goalkeeper_trajectory(
    candidates: List[GoalkeeperPoint],
    ball_traj: List[BallPoint],
    prepend_frames: int = 5,
) -> Tuple[List[GoalkeeperPoint], str]:
    """Select one goalkeeper trajectory aligned to ball timestamps.

    Result uses the exact same timestamp grid as ball_traj.

    Among all candidate sensors, this picks the one that is most likely the
    goalkeeper of the defending team for this shot. The selection is based on:

    1. The sensor must be on the same side of the field as the shot direction.
    2. The sensor must have points within the ball trajectory's time range.
    3. Among the remaining sensors, the best one is chosen by a score that
       prefers: more points in range, being farther from the center line
       (i.e. closer to the goal), and being closer to the center of the goal.

    Args:
        candidates: Goalkeeper candidate points (already filtered to goal areas).
        ball_traj: The ball trajectory to align the goalkeeper to.
        prepend_frames: Unused parameter (kept for API compatibility).

    Returns:
        A tuple ``(aligned_goalkeeper_points, sensor_id)``. If no suitable
        goalkeeper is found, returns ``([], "")``.
    """
    if not candidates or not ball_traj:
        return [], ""

    # Time range of the ball trajectory we must cover.
    start_dt = ball_traj[0].local_dt
    end_dt = ball_traj[-1].local_dt
    # Which side of the field is the shot heading to?
    same_side_sign = _detect_shot_direction(ball_traj)

    # Group candidate points by tracked sensor.
    by_sensor: Dict[str, List[GoalkeeperPoint]] = {}
    for p in candidates:
        by_sensor.setdefault(p.sensor_id, []).append(p)

    best_sensor = ""
    best_segment: List[GoalkeeperPoint] = []
    best_score: Optional[Tuple[int, float, float]] = None

    for sensor_id, points in by_sensor.items():
        # A goalkeeper on the side the ball is NOT heading to cannot be the
        # defending goalkeeper for this shot.
        sensor_points = [
            p
            for p in points
            if int(math.copysign(1, p.x)) == same_side_sign
        ]
        if not sensor_points:
            continue

        # Indices of this sensor's points that fall inside the ball's time range.
        in_range_indices = [
            i for i, p in enumerate(sensor_points) if start_dt <= p.local_dt <= end_dt
        ]
        if not in_range_indices:
            continue

        # Take the contiguous segment of this sensor covering the ball's window.
        first_in_range = in_range_indices[0]
        last_in_range = in_range_indices[-1]
        segment = sensor_points[first_in_range : last_in_range + 1]
        if not segment:
            continue

        # Score the segment: prefer more points, being far from the center line
        # (mean_abs_x large -> close to goal), and being near the goal center
        # (mean_abs_y small -> -mean_abs_y large).
        mean_abs_x = sum(abs(p.x) for p in segment) / len(segment)
        mean_abs_y = sum(abs(p.y) for p in segment) / len(segment)
        score = (len(segment), mean_abs_x, -mean_abs_y)

        # Keep the best-scoring sensor and align it to the ball timestamps.
        if best_score is None or score > best_score:
            best_score = score
            best_sensor = sensor_id
            best_segment = _align_goalkeeper_to_ball_timestamps(segment, ball_traj, sensor_id)

    return best_segment, best_sensor


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def first_idx_at_or_after(points: List[BallPoint], dt: datetime) -> Optional[int]:
    """Return the index of the first point with local_dt >= dt, else None."""
    for i, p in enumerate(points):
        if p.local_dt >= dt:
            return i
    return None


def first_idx_at_or_before(points: List[BallPoint], dt: datetime) -> Optional[int]:
    """Return the index of the last point with local_dt <= dt, else None."""
    for i in range(len(points) - 1, -1, -1):
        if points[i].local_dt <= dt:
            return i
    return None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_point(point: BallPoint) -> str:
    """Serialize a single BallPoint to a compact JSON string.

    NaN speed/accel values are converted to null so the JSON stays valid.
    """
    return json.dumps(
        {
            "t_local": point.local_dt.isoformat(timespec="milliseconds"),
            "ts_ms": point.ts_ms,
            "x": point.x,
            "y": point.y,
            "z": point.z,
            "v": None if math.isnan(point.speed) else point.speed,
            "a": None if math.isnan(point.accel) else point.accel,
            "dir": point.direction,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Release point detection (heuristic helpers)
# ---------------------------------------------------------------------------

# Finds first index of given points, where x ordinate is behind either goal line
def find_goalline_crossing_idx(points: List[BallPoint], start_idx: int) -> Optional[int]:
    """Return the first index at/after start_idx where the ball crosses a goal line.

    The goal lines are at x = +20 and x = -20. Returns None if the ball never
    crosses either line within the given points.
    """
    for i in range(start_idx, len(points)):
        p = points[i]
        if p.x > 20.0 or p.x < -20.0:
            return i
    return None


# Finds the release point earlier
def detect_release_for_short_distance(
    points: List[BallPoint], start_idx: int, back_window_ms: int = 3000
) -> Optional[int]:
    """Detect an earlier release point when the shot starts very close to the goal.

    When the recorded shot start is already very close to the goal (distance
    < ~7 m), the actual release likely happened before the recorded start time.
    This function looks back up to ``back_window_ms`` before the start point,
    finds the peak acceleration in that window (the thrower accelerating the
    ball), and then walks backwards while acceleration is non-decreasing to
    locate the true release.

    Args:
        points: Full ball trajectory.
        start_idx: Index of the recorded shot start.
        back_window_ms: How far back (in ms) to search for the release.

    Returns:
        Index of the detected release point, or None if it could not be found.
    """
    start_dt = points[start_idx].local_dt
    min_dt = start_dt - timedelta(milliseconds=back_window_ms)

    candidates = [i for i in range(len(points)) if min_dt <= points[i].local_dt <= start_dt]
    if not candidates:
        return None

    candidates_with_accel = [i for i in candidates if not math.isnan(points[i].accel)]
    if not candidates_with_accel:
        return None

    max_accel = max(points[i].accel for i in candidates_with_accel)
    eps = 1e-6
    max_idxs = [i for i in candidates_with_accel if abs(points[i].accel - max_accel) <= eps]
    if not max_idxs:
        return None

    # Prefer the final sample when the peak spans multiple frames.
    peak_idx = max(max_idxs)

    # Walk backwards while the previous acceleration is >= the current one;
    # this traces the acceleration ramp-up back to the release.
    release_idx = peak_idx
    while release_idx > 0:
        prev_i = release_idx - 1
        prev_a = points[prev_i].accel
        cur_a = points[release_idx].accel
        if math.isnan(prev_a) or math.isnan(cur_a):
            break
        if prev_a >= cur_a:
            release_idx = prev_i
            continue
        break

    return release_idx


def has_x_direction_reversal(points: List[BallPoint], start_idx: int) -> bool:
    """Return True if the ball reverses its x-direction after start_idx.

    A real penalty throw should move monotonically towards the goal in x.
    If the ball changes x-direction after the candidate release point, the
    candidate is probably not a valid release point.
    """
    if start_idx >= len(points) - 1:
        return False

    start_x = points[start_idx].x
    direction = None

    for i in range(start_idx + 1, len(points)):
        x = points[i].x
        if x == start_x:
            continue

        current_direction = 1 if x > start_x else -1

        if direction is None:
            direction = current_direction
        elif direction != current_direction:
            return True

    return False


def is_plausible_release_point(
    points: List[BallPoint],
    start_idx: int,
    distance: Optional[float],
) -> bool:
    """Heuristic check whether the given start index is a plausible release point.

    A release point is considered plausible if:
    - The reported shot distance is known.
    - The distance is within the 7 m penalty range (6.5..7.5 m).
    - The ball does not reverse its x-direction afterwards.

    Args:
        points: Full ball trajectory.
        start_idx: Candidate release index.
        distance: Reported shot distance in meters (from penalties.csv).

    Returns:
        True if the candidate is plausible, False otherwise.
    """
    if start_idx >= len(points):
        return False

    if distance is None:
        return False

    if not (6.5 <= distance <= 7.5):
        return False

    if has_x_direction_reversal(points, start_idx):
        return False

    return True


def adjust_start_idx_by_distance(
    points: List[BallPoint],
    start_idx: int,
    distance: Optional[float],
) -> Tuple[int, dict[str, str], Optional[int]]:
    """Adjust the trajectory start index based on the reported shot distance.

    The recorded shot start time may not coincide with the actual release. This
    function corrects the start index depending on how far the shot was:

    - If the distance is very short (< 6.95 m), the release likely happened
      before the recorded start -> search backwards for the release and move
      the start 1 s before it.
    - If the distance is long (> 8 m), the thrower likely started farther back
      -> move the start to the first point within the 12..14 m x-gate.
    - Otherwise the start is left unchanged.

    Args:
        points: Full ball trajectory.
        start_idx: Current start index.
        distance: Reported shot distance in meters (may be None).

    Returns:
        A tuple ``(corrected_start_idx, flags, release_idx)`` where ``flags``
        is a dict describing what correction was applied (or why none was) and
        ``release_idx`` is the detected release index if one was found.
    """
    flags: dict[str, str] = {}
    release_idx: Optional[int] = None

    if distance is None:
        flags["distance_check"] = "missing_distance"
        return start_idx, flags, release_idx

    if distance < 6.95:
        rel_idx = detect_release_for_short_distance(points, start_idx)
        if rel_idx is None:
            flags["short_distance_correction"] = "no_release_detected"
            return start_idx, flags, release_idx

        release_idx = rel_idx
        # Keep one second of context before the detected release.
        corrected_start_dt = points[release_idx].local_dt - timedelta(seconds=1)
        corrected_start_idx = first_idx_at_or_after(points, corrected_start_dt)
        if corrected_start_idx is not None:
            flags["short_distance_correction"] = "applied"
            return corrected_start_idx, flags, release_idx

        flags["short_distance_correction"] = "failed_index_lookup"
        return start_idx, flags, release_idx

    if distance > 8:
        # For long shots, move the start to the first point inside the
        # 12..14 m x-gate (the typical run-up zone).
        for i in range(start_idx, len(points)):
            x = points[i].x
            if (12.0 <= x <= 14.0) or (-14.0 <= x <= -12.0):
                flags["long_distance_correction"] = "applied"
                return i, flags, release_idx
        flags["long_distance_correction"] = "no_x_gate_match"
        return start_idx, flags, release_idx

    # Distance is within the normal range -> no correction needed.
    flags["distance_check"] = "in_range"
    return start_idx, flags, release_idx


def serialize_trajectory(points: List[BallPoint]) -> str:
    """Serialize a list of BallPoint to a compact JSON array string."""
    payload = []
    for p in points:
        payload.append(
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
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def serialize_goalkeeper_trajectory(points: List[GoalkeeperPoint]) -> str:
    """Serialize a list of GoalkeeperPoint to a compact JSON array string."""
    payload = []
    for p in points:
        payload.append(
            {
                "t_local": p.local_dt.isoformat(timespec="milliseconds"),
                "ts_ms": p.ts_ms,
                "x": p.x,
                "y": p.y,
                "z": p.z,
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Physics-based release point detection
# ---------------------------------------------------------------------------

def _infer_goal_sign(points: List[BallPoint], start_idx: int) -> int:
    """Infer which goal the shot is heading towards.

    Looks at the first few points after start_idx and returns +1 if the ball is
    moving towards the positive-x goal (x=+20) or -1 for the negative-x goal.
    """
    if not points:
        return 1

    start_idx = max(0, min(start_idx, len(points) - 1))
    tail = points[start_idx : min(len(points), start_idx + 8)]
    if not tail:
        tail = points

    avg_x = sum(p.x for p in tail) / len(tail)
    return 1 if avg_x >= 0 else -1


def _goal_x_from_sign(goal_sign: int) -> float:
    """Return the x coordinate of the goal line for a given goal sign."""
    return 20.0 if goal_sign >= 0 else -20.0


def _distance_to_goal_line(point: BallPoint, goal_sign: int) -> float:
    """Return the horizontal distance from a point to the goal line."""
    return abs(_goal_x_from_sign(goal_sign) - point.x)


def find_goal_cutoff_idx(
    points: List[BallPoint],
    start_idx: int,
    goal_distance_limit_m: float = 4.2,
) -> Optional[int]:
    """Return the first index where the ball gets within goal_distance_limit_m of the goal.

    This marks the point where the ball is so close to the goal that it can no
    longer be considered "in flight" for release detection purposes. Points
    after this index are excluded from the release search.
    """
    goal_sign = _infer_goal_sign(points, start_idx)
    for i in range(start_idx, len(points)):
        if _distance_to_goal_line(points[i], goal_sign) <= goal_distance_limit_m:
            return i
    return None


def find_ground_contact_idx(
    points: List[BallPoint],
    start_idx: int,
    ground_z_threshold_m: float = 0.18,
) -> Optional[int]:
    """Return the first index where the ball appears to touch the ground.

    A ground contact is detected at index i when the z value is below the
    threshold and forms a local minimum (z decreases then increases). This is
    used to stop the release search before the ball bounces.
    """
    if start_idx >= len(points) - 2:
        return None

    for i in range(start_idx + 1, len(points) - 1):
        prev_z = points[i - 1].z
        cur_z = points[i].z
        next_z = points[i + 1].z

        if cur_z > ground_z_threshold_m:
            continue

        # Local minimum in z -> ball touched the ground here.
        if cur_z <= prev_z and next_z >= cur_z:
            return i

    return None


def _finite_difference_velocity(points: List[BallPoint]) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Estimate velocity between consecutive points via finite differences.

    Returns a tuple ``(velocities, times)`` where each velocity is a
    (vx, vy, vz) tuple computed between two consecutive points, and ``times``
    holds the midpoint time of each interval (relative to the first point).
    """
    if len(points) < 2:
        return [], []

    velocities: List[Tuple[float, float, float]] = []
    times: List[float] = []
    for i in range(len(points) - 1):
        dt = (points[i + 1].local_dt - points[i].local_dt).total_seconds()
        if dt <= 0:
            continue
        velocities.append(
            (
                (points[i + 1].x - points[i].x) / dt,
                (points[i + 1].y - points[i].y) / dt,
                (points[i + 1].z - points[i].z) / dt,
            )
        )
        # Midpoint time of the interval, relative to the first point.
        times.append((points[i + 1].local_dt - points[0].local_dt).total_seconds() - dt / 2.0)

    if not velocities:
        return [], []

    return velocities, times


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Solve a square linear system A*x = b using Gaussian elimination.

    Returns the solution vector. If the matrix is (near) singular, returns a
    zero vector instead of raising.
    """
    size = len(matrix)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for pivot_index in range(size):
        # Partial pivoting: pick the row with the largest absolute pivot.
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        pivot_value = augmented[pivot_row][pivot_index]
        if abs(pivot_value) < 1e-12:
            return [0.0 for _ in range(size)]
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        # Normalize the pivot row.
        pivot_value = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot_value

        # Eliminate the pivot column from all other rows.
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0:
                continue
            for column in range(pivot_index, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot_index][column]

    return [augmented[index][size] for index in range(size)]


def _least_squares_fit(design_rows: List[List[float]], values: List[float]) -> List[float]:
    """Solve a least-squares problem via the normal equations (A^T A x = A^T b).

    Args:
        design_rows: Design matrix rows (each row is one observation).
        values: Observed target values b.

    Returns:
        The fitted coefficient vector, or [] if there are no observations.
    """
    if not design_rows:
        return []

    column_count = len(design_rows[0])
    ata = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
    atb = [0.0 for _ in range(column_count)]

    for row, value in zip(design_rows, values):
        for i in range(column_count):
            atb[i] += row[i] * value
            for j in range(column_count):
                ata[i][j] += row[i] * row[j]

    return _solve_linear_system(ata, atb)


def _fit_projectile_segment(
    segment: List[BallPoint],
    goal_sign: int,
) -> Dict[str, Any]:
    """Fit a ballistic (projectile) model to a trajectory segment.

    A free-flying ball follows a parabola in z (constant gravity) and straight
    lines in x and y. This function fits:
    - x(t) = a0 + a1*t            (linear)
    - y(t) = b0 + b1*t            (linear)
    - z(t) = c0 + c1*t + c2*t^2   (quadratic, c2 ~ -g/2)

    It then scores how well the segment matches this model and returns the
    fitted release position/velocity plus several quality sub-scores.

    Args:
        segment: The trajectory points to fit (candidate release point first).
        candidate_idx: Index of the candidate release point (used for timing).
        goal_sign: +1 or -1 indicating which goal the shot heads towards.

    Returns:
        A dict with validity flag, RMSE values, fitted release position and
        velocity, and the projectile/kinematic/distance sub-scores.
    """
    if len(segment) < 3:
        return {
            "valid": False,
            "reason": "too_short",
            "position_rmse": float("inf"),
            "velocity_rmse": float("inf"),
            "gravity_error": float("inf"),
            "release_position": None,
            "release_velocity": None,
            "predicted_release_speed": float("nan"),
        }

    # Time in seconds relative to the candidate release point.
    candidate_dt = segment[0].local_dt
    times = [(p.local_dt - candidate_dt).total_seconds() for p in segment]
    x = [p.x for p in segment]
    y = [p.y for p in segment]
    z = [p.z for p in segment]

    # Design matrices: linear for x/y, quadratic for z.
    design_linear = [[1.0, t] for t in times]
    design_quadratic = [[1.0, t, t * t] for t in times]

    coef_x = _least_squares_fit(design_linear, x)
    coef_y = _least_squares_fit(design_linear, y)
    coef_z = _least_squares_fit(design_quadratic, z)

    # Reconstruct the fitted positions to compute the position RMSE.
    pred_x = [coef_x[0] + coef_x[1] * t for t in times]
    pred_y = [coef_y[0] + coef_y[1] * t for t in times]
    pred_z = [coef_z[0] + coef_z[1] * t + coef_z[2] * t * t for t in times]
    position_rmse = math.sqrt(
        sum((x_i - px) ** 2 + (y_i - py) ** 2 + (z_i - pz) ** 2 for x_i, px, y_i, py, z_i, pz in zip(x, pred_x, y, pred_y, z, pred_z))
        / len(segment)
    )

    # Compare the model's velocity to the observed finite-difference velocity.
    observed_velocity, velocity_times = _finite_difference_velocity(segment)
    if len(observed_velocity) > 0:
        pred_velocity = [
            (coef_x[1], coef_y[1], coef_z[1] + 2.0 * coef_z[2] * t)
            for t in velocity_times
        ]
        velocity_rmse = math.sqrt(
            sum(
                (obs_x - pred_x_v) ** 2 + (obs_y - pred_y_v) ** 2 + (obs_z - pred_z_v) ** 2
                for (obs_x, obs_y, obs_z), (pred_x_v, pred_y_v, pred_z_v) in zip(observed_velocity, pred_velocity)
            )
            / len(observed_velocity)
        )
    else:
        velocity_rmse = float("inf")

    # For a free-flying ball, 2*c2 should equal -g (9.81 m/s^2).
    gravity_error = float(abs(2.0 * coef_z[2] + 9.81))

    # Release velocity/position come from the fitted coefficients at t=0.
    release_velocity = {
        "vx": float(coef_x[1]),
        "vy": float(coef_y[1]),
        "vz": float(coef_z[1]),
        "speed": float(math.sqrt(coef_x[1] ** 2 + coef_y[1] ** 2 + coef_z[1] ** 2)),
    }
    release_position = {
        "x": float(coef_x[0]),
        "y": float(coef_y[0]),
        "z": float(coef_z[0]),
    }

    release_distance_to_goal = abs(_goal_x_from_sign(goal_sign) - segment[0].x)

    # Convert errors into 0..1 scores (smaller error -> higher score).
    position_score = 1.0 / (1.0 + position_rmse / 0.18)
    velocity_score = 1.0 / (1.0 + velocity_rmse / 1.75) if math.isfinite(velocity_rmse) else 0.0
    gravity_score = 1.0 / (1.0 + gravity_error / 4.0)
    projectile_score = 0.7 * position_score + 0.2 * velocity_score + 0.1 * gravity_score

    # Reward releases at least 6.5 m from the goal (i.e. near/behind the 7 m line).
    distance_score = 1.0 if release_distance_to_goal >= 6.5 else max(0.0, 1.0 - ((6.5 - release_distance_to_goal) / 1.5) ** 2)
    # Reward having enough points after the release to make the fit reliable.
    coverage_score = min(1.0, max(0.0, (len(segment) - 2) / 8.0))
    kinematic_score = 0.6 * velocity_score + 0.4 * gravity_score

    return {
        "valid": True,
        "position_rmse": position_rmse,
        "velocity_rmse": velocity_rmse,
        "gravity_error": gravity_error,
        "release_position": release_position,
        "release_velocity": release_velocity,
        "release_distance_to_goal": release_distance_to_goal,
        "projectile_score": projectile_score * coverage_score,
        "kinematic_score": kinematic_score * coverage_score,
        "distance_score": distance_score,
        "coverage_score": coverage_score,
        "predicted_release_speed": release_velocity["speed"],
    }


def _score_possession_context(
    candidates: List[PossessionContextPoint],
    release_dt: datetime,
    window_before_ms: int = 400,
    window_after_ms: int = 300,
) -> Dict[str, Any]:
    """Score how well the possession context supports a candidate release time.

    Around the release moment, the thrower should be the one possessing the
    ball just before release, and should no longer possess it right after.
    This function finds the closest possession point before the release
    ("pre") and the closest one after the release ("post") and derives a
    ``core_score`` from three aspects:

    - temporal_score: how close the pre point is to the release time.
    - transition_score: whether possession changes between pre and post
      (a change suggests the ball was released).
    - motion_score: how fast/accelerated the pre point is (a thrower is
      usually moving/accelerating).

    Args:
        candidates: Possession-context points (chronologically sorted).
        release_dt: Candidate release timestamp.
        window_before_ms: Max look-back window for the "pre" point.
        window_after_ms: Max look-ahead window for the "post" point.

    Returns:
        A dict with ``core_score``, ``pre_context``, ``post_context`` and
        ``context_gap_ms`` (time between release and the pre point).
    """
    if not candidates:
        return {
            "core_score": 0.0,
            "pre_context": None,
            "post_context": None,
            "context_gap_ms": None,
            "possession_transition": None,
        }

    # Collect candidate points before and after the release time.
    before: List[Tuple[int, PossessionContextPoint]] = []
    after: List[Tuple[int, PossessionContextPoint]] = []
    for point in candidates:
        delta_ms = int((release_dt - point.local_dt).total_seconds() * 1000)
        if 0 <= delta_ms <= window_before_ms:
            before.append((delta_ms, point))
        elif 0 <= -delta_ms <= window_after_ms:
            after.append((-delta_ms, point))

    occupied_before = [item for item in before if item[1].possession_id]
    # Closest occupied point before release if available, otherwise the nearest
    # point before release of any kind.
    pre = min(occupied_before, key=lambda item: item[0])[1] if occupied_before else (min(before, key=lambda item: item[0])[1] if before else None)

    # Prefer the same tracked sensor after the release so we can observe the
    # possession dropping to empty for that player.
    post_same_sensor = None
    if pre is not None:
        same_sensor_after_empty = [item for item in after if item[1].sensor_id == pre.sensor_id and not item[1].possession_id]
        same_sensor_after_any = [item for item in after if item[1].sensor_id == pre.sensor_id]
        if same_sensor_after_empty:
            post_same_sensor = min(same_sensor_after_empty, key=lambda item: item[0])[1]
        elif same_sensor_after_any:
            post_same_sensor = min(same_sensor_after_any, key=lambda item: item[0])[1]
    post = post_same_sensor if post_same_sensor is not None else (min(after, key=lambda item: item[0])[1] if after else None)

    # Temporal score: 1.0 if the pre point is exactly at release, decaying to 0
    # at the edge of the window.
    temporal_score = 0.0
    if pre is not None:
        temporal_score = max(0.0, 1.0 - min(1.0, abs((release_dt - pre.local_dt).total_seconds()) / (window_before_ms / 1000.0)))

    # Transition score: if there is no post point, the ball was likely released
    # (nobody possesses it anymore). If the possessor changed, that is also
    # strong evidence. If the same person still possesses the ball, weak.
    transition_score = 0.0
    if pre is not None:
        if post is None:
            transition_score = 1.0
        elif post.sensor_id == pre.sensor_id and not post.possession_id:
            transition_score = 1.0
        elif pre.possession_id and post.possession_id != pre.possession_id:
            transition_score = 0.85
        else:
            transition_score = 0.35

    # Motion score: a thrower is usually moving/accelerating before release.
    motion_score = 0.0
    if pre is not None:
        speed_term = 0.0 if math.isnan(pre.speed) else min(1.0, pre.speed / 8.0)
        accel_term = 0.0 if math.isnan(pre.accel) else min(1.0, abs(pre.accel) / 20.0)
        motion_score = 0.6 * speed_term + 0.4 * accel_term

    core_score = 0.45 * temporal_score + 0.35 * transition_score + 0.20 * motion_score

    possession_transition = None
    if pre is not None:
        possession_transition = {
            "pre": {
                "t_local": pre.local_dt.isoformat(timespec="milliseconds"),
                "sensor_id": pre.sensor_id,
                "full_name": pre.full_name,
                "group_name": pre.group_name,
                "possession_id": pre.possession_id,
            },
            "post": None
            if post is None
            else {
                "t_local": post.local_dt.isoformat(timespec="milliseconds"),
                "sensor_id": post.sensor_id,
                "full_name": post.full_name,
                "group_name": post.group_name,
                "possession_id": post.possession_id,
            },
            "delta_ms": None if post is None else int((post.local_dt - pre.local_dt).total_seconds() * 1000),
            "transition_to_empty": bool(pre.possession_id and post is not None and post.sensor_id == pre.sensor_id and not post.possession_id),
        }
    return {
        "core_score": core_score,
        "pre_context": None
        if pre is None
        else {
            "t_local": pre.local_dt.isoformat(timespec="milliseconds"),
            "full_name": pre.full_name,
            "group_name": pre.group_name,
            "possession_id": pre.possession_id,
            "speed": None if math.isnan(pre.speed) else pre.speed,
            "accel": None if math.isnan(pre.accel) else pre.accel,
        },
        "post_context": None
        if post is None
        else {
            "t_local": post.local_dt.isoformat(timespec="milliseconds"),
            "full_name": post.full_name,
            "group_name": post.group_name,
            "possession_id": post.possession_id,
            "speed": None if math.isnan(post.speed) else post.speed,
            "accel": None if math.isnan(post.accel) else post.accel,
        },
        "context_gap_ms": None if pre is None else int((release_dt - pre.local_dt).total_seconds() * 1000),
        "possession_transition": possession_transition,
    }


def _build_release_search_bounds(
    points: List[BallPoint],
    start_idx: int,
    goal_distance_limit_m: float,
    ground_z_threshold_m: float,
    min_post_points: int,
) -> Dict[str, Any]:
    """Compute the indices that bound the candidate search window."""
    goal_cutoff_idx = find_goal_cutoff_idx(points, start_idx, goal_distance_limit_m=goal_distance_limit_m)
    ground_contact_idx = find_ground_contact_idx(points, start_idx, ground_z_threshold_m=ground_z_threshold_m)

    search_end_idx = len(points) - 1
    if goal_cutoff_idx is not None:
        search_end_idx = min(search_end_idx, max(start_idx, goal_cutoff_idx - 1))
    if ground_contact_idx is not None:
        search_end_idx = min(search_end_idx, ground_contact_idx)

    candidate_start_idx = start_idx
    candidate_end_idx = max(candidate_start_idx, search_end_idx - min_post_points + 1)

    return {
        "goal_cutoff_idx": goal_cutoff_idx,
        "ground_contact_idx": ground_contact_idx,
        "search_end_idx": search_end_idx,
        "candidate_start_idx": candidate_start_idx,
        "candidate_end_idx": candidate_end_idx,
        "goal_sign": _infer_goal_sign(points, start_idx),
    }


def _score_release_candidate(
    points: List[BallPoint],
    idx: int,
    search_end_idx: int,
    goal_sign: int,
    possession_candidates: List[PossessionContextPoint],
) -> Optional[Dict[str, Any]]:
    """Score one candidate release index against the post-release segment."""
    segment = points[idx : search_end_idx + 1]
    fit = _fit_projectile_segment(segment, goal_sign)
    if not fit.get("valid"):
        return None

    release_dt = points[idx].local_dt
    core_context = _score_possession_context(possession_candidates, release_dt)

    composite_score = (
        0.60 * fit["projectile_score"]
        + 0.15 * fit["kinematic_score"]
        + 0.10 * core_context["core_score"]
        + 0.15 * fit["distance_score"]
    )

    return {
        "idx": idx,
        "score": composite_score,
        "projectile_score": fit["projectile_score"],
        "kinematic_score": fit["kinematic_score"],
        "core_score": core_context["core_score"],
        "distance_score": fit["distance_score"],
        "position_rmse": fit["position_rmse"],
        "velocity_rmse": fit["velocity_rmse"],
        "gravity_error": fit["gravity_error"],
        "release_position": fit["release_position"],
        "release_velocity": fit["release_velocity"],
        "release_distance_to_goal": fit["release_distance_to_goal"],
        "post_points": len(segment),
        "pre_context": core_context["pre_context"],
        "post_context": core_context["post_context"],
        "possession_transition": core_context["possession_transition"],
    }


def _confidence_from_candidates(
    best_candidate: Dict[str, Any],
    second_best_score: float,
    candidate_start_idx: int,
    candidate_end_idx: int,
) -> float:
    """Convert the top-two candidate scores into a 0..1 confidence value."""
    best_gap = best_candidate["score"] - second_best_score
    edge_penalty = 0.1 if best_candidate["idx"] in {candidate_start_idx, candidate_end_idx} else 0.0

    return max(
        0.0,
        min(
            1.0,
            0.45 * best_candidate["projectile_score"]
            + 0.20 * min(1.0, best_gap / 0.15)
            + 0.15 * min(1.0, best_candidate["post_points"] / 10.0)
            + 0.10 * best_candidate["distance_score"]
            + 0.10 * best_candidate["core_score"]
            - edge_penalty,
        ),
    )


def detect_physics_based_release_point(
    points: List[BallPoint],
    start_idx: int,
    possession_candidates: Optional[List[PossessionContextPoint]] = None,
    goal_distance_limit_m: float = 4.2,
    ground_z_threshold_m: float = 0.18,
    min_post_points: int = 5,
) -> ReleaseDetectionResult:
    """Detect the release point by fitting projectile models to candidate windows.

    This is the main physics-based release detector. It works as follows:

    1. Determine the search range: from ``start_idx`` up to just before the
       goal cutoff and/or ground contact.
    2. For every candidate index in that range, fit a projectile model to the
       remaining segment and compute a composite score combining projectile
       fit, kinematic fit, possession context, and distance-to-goal.
    3. Pick the candidate with the highest composite score and compute a
       confidence value based on the score, the gap to the second-best
       candidate, the number of post points, and an edge penalty.

    Args:
        points: Ball trajectory to search in.
        start_idx: First index to consider as a release candidate.
        possession_candidates: Optional possession points for context scoring.
        goal_distance_limit_m: Stop search when ball is this close to the goal.
        ground_z_threshold_m: Z threshold for ground-contact detection.
        min_post_points: Minimum number of points required after a candidate
            for it to be considered (ensures enough data for the fit).

    Returns:
        A ReleaseDetectionResult with the best candidate (or None if no valid
        candidate was found).
    """
    if not points or start_idx >= len(points):
        return ReleaseDetectionResult(
            release_idx=None,
            candidate_window=(0, -1),
            goal_cutoff_idx=None,
            ground_contact_idx=None,
            release_point=None,
            release_velocity=None,
            release_distance_to_goal=None,
            release_score=0.0,
            projectile_score=0.0,
            kinematic_score=0.0,
            core_score=0.0,
            distance_score=0.0,
            confidence=0.0,
            diagnostics={"reason": "empty_points"},
        )

    bounds = _build_release_search_bounds(
        points,
        start_idx,
        goal_distance_limit_m=goal_distance_limit_m,
        ground_z_threshold_m=ground_z_threshold_m,
        min_post_points=min_post_points,
    )
    goal_cutoff_idx = bounds["goal_cutoff_idx"]
    ground_contact_idx = bounds["ground_contact_idx"]
    search_end_idx = bounds["search_end_idx"]
    candidate_start_idx = bounds["candidate_start_idx"]
    candidate_end_idx = bounds["candidate_end_idx"]
    goal_sign = bounds["goal_sign"]
    possession_candidates = possession_candidates or []

    candidate_scores: List[Dict[str, Any]] = []
    best_candidate: Optional[Dict[str, Any]] = None

    # Try every index in the candidate window as a potential release point.
    for idx in range(candidate_start_idx, candidate_end_idx + 1):
        scored_candidate = _score_release_candidate(
            points,
            idx,
            search_end_idx,
            goal_sign,
            possession_candidates,
        )
        if scored_candidate is None:
            continue
        candidate_scores.append(scored_candidate)
        if best_candidate is None or scored_candidate["score"] > best_candidate["score"]:
            best_candidate = scored_candidate

    if best_candidate is None:
        return ReleaseDetectionResult(
            release_idx=None,
            candidate_window=(candidate_start_idx, candidate_end_idx),
            goal_cutoff_idx=goal_cutoff_idx,
            ground_contact_idx=ground_contact_idx,
            release_point=None,
            release_velocity=None,
            release_distance_to_goal=None,
            release_score=0.0,
            projectile_score=0.0,
            kinematic_score=0.0,
            core_score=0.0,
            distance_score=0.0,
            confidence=0.0,
            diagnostics={
                "reason": "no_valid_candidates",
                "candidate_scores": candidate_scores,
            },
        )

    # Confidence: combine the best score, the gap to the second-best candidate,
    # the number of post points, distance score, and context score. Candidates
    # at the very edge of the search window get a penalty because the true
    # release may lie just outside the window.
    sorted_candidates = sorted(candidate_scores, key=lambda item: item["score"], reverse=True)
    second_best_score = sorted_candidates[1]["score"] if len(sorted_candidates) > 1 else 0.0
    best_gap = best_candidate["score"] - second_best_score
    confidence = _confidence_from_candidates(
        best_candidate,
        second_best_score,
        candidate_start_idx,
        candidate_end_idx,
    )

    diagnostics: Dict[str, Any] = {
        "candidate_scores": candidate_scores,
        "best_candidate_rank_gap": best_gap,
        "search_end_idx": search_end_idx,
        "goal_sign": goal_sign,
        "possession_transition": best_candidate.get("possession_transition"),
    }

    return ReleaseDetectionResult(
        release_idx=best_candidate["idx"],
        candidate_window=(candidate_start_idx, candidate_end_idx),
        goal_cutoff_idx=goal_cutoff_idx,
        ground_contact_idx=ground_contact_idx,
        release_point=points[best_candidate["idx"]],
        release_velocity=best_candidate["release_velocity"],
        release_distance_to_goal=best_candidate["release_distance_to_goal"],
        release_score=best_candidate["score"],
        projectile_score=best_candidate["projectile_score"],
        kinematic_score=best_candidate["kinematic_score"],
        core_score=best_candidate["core_score"],
        distance_score=best_candidate["distance_score"],
        confidence=confidence,
        diagnostics=diagnostics,
    )


def select_release_point(points: List[BallPoint]) -> Tuple[BallPoint, str]:
    """Select a release point using a simple heuristic (legacy method).

    The heuristic works as follows:
    1. Restrict to points within the 6.5..7.5 m release zone around the goal.
    2. Among those, pick the point with the highest acceleration (the thrower
       accelerating the ball just before release).
    3. If no such point exists, fall back to the global max acceleration, then
       global max speed, then the first point.

    Args:
        points: Ball trajectory to search in.

    Returns:
        A tuple ``(release_point, flag)`` where ``flag`` describes which
        strategy was used.
    """
    if not points:
        raise ValueError("select_release_point requires at least one trajectory point")

    start_x = points[0].x if points else 0.0
    goal_x = 20.0 if start_x > 0 else -20.0

    # Indices of points within the 7 m penalty zone around the goal.
    release_zone_indices = []
    for i, p in enumerate(points):
        dx = p.x - goal_x
        dy = p.y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        if 6.5 <= distance_to_goal <= 7.5:
            release_zone_indices.append(i)

    # Prefer the point with the highest acceleration inside the release zone.
    finite_accel_indices = [i for i in release_zone_indices if not math.isnan(points[i].accel)]
    if finite_accel_indices:
        rel_local_i = max(finite_accel_indices, key=lambda i: points[i].accel)
        return points[rel_local_i], "release_point:max_a"

    # Fall back to the global acceleration maximum when the release zone has
    # no finite acceleration samples.
    finite_accel_indices = [i for i, p in enumerate(points) if not math.isnan(p.accel)]
    if finite_accel_indices:
        rel_local_i = max(finite_accel_indices, key=lambda i: points[i].accel)
        return points[rel_local_i], "release_point:max_a_fallback_global"

    # If acceleration is unavailable, use the global speed maximum.
    finite_speed_indices = [i for i, p in enumerate(points) if not math.isnan(p.speed)]
    if finite_speed_indices:
        rel_local_i = max(finite_speed_indices, key=lambda i: points[i].speed)
        return points[rel_local_i], "release_point:max_v_fallback"

    return points[0], "release_point:first_point_fallback"
