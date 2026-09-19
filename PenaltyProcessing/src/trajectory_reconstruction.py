#!/usr/bin/env python3
"""Geometry-preserving reconstruction of matched League trajectories.

Reconstruction performs exactly one transformation of measured positions: a constant
translation from the League trajectory's PoR to the Mocap PoR.  Upsampling
only evaluates a shape-preserving interpolant through those translated points.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


XYZ = ("x", "y", "z")
WEIGHT_GROUP_NAMES = (
    "release_speed", "release_angles", "release_height", "relative_trajectory",
    "velocity_evolution", "trajectory_angles", "acceleration", "absolute_por",
)


@dataclass(frozen=True)
class BounceWindow:
    """Local collision-reconstruction window around an observed z minimum."""

    minimum_idx: int
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class BounceEvent:
    time_ms: float
    x: float
    y: float
    z: float
    vertical_fit_rmse: float
    vz_in_m_s: float
    vz_out_m_s: float
    fit_valid: bool
    boundary_start_ms: float
    boundary_end_ms: float
    boundary_vz_in_m_s: float
    boundary_vz_out_m_s: float
    reconstruction_rmse_m: float
    reconstruction_max_error_m: float


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    # trajectory_json is legitimately much larger than Python's default
    # 128 KiB per-field limit. Use the largest value accepted by this build.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        # Detect from the header only. Sampling data rows is unreliable here:
        # trajectory_json can be several KB and contains hundreds of commas,
        # which makes csv.Sniffer misclassify a semicolon-delimited raw file.
        header = handle.readline()
        handle.seek(0)
        delimiter = max((";", ",", "\t"), key=header.count)
        if header.count(delimiter) == 0:
            raise ValueError(f"Could not detect a supported CSV delimiter in {path}")
        return list(csv.DictReader(handle, delimiter=delimiter)), delimiter


def load_raw_throw(csv_path: str | Path, throw_id: object) -> dict[str, str]:
    """Load one throw from a raw representation CSV without coercing its ID."""
    wanted = str(throw_id)
    rows, _ = _read_csv(Path(csv_path))
    for row in rows:
        if str(row.get("throw_id", "")) == wanted:
            return row
    raise KeyError(f"throw_id {wanted!r} not found in {csv_path}")


def parse_trajectory_json(row_or_json: Mapping[str, Any] | str) -> list[dict[str, Any]]:
    """Parse, validate, time-sort, and annotate measured trajectory samples."""
    raw = row_or_json.get("trajectory_json", "[]") if isinstance(row_or_json, Mapping) else row_or_json
    decoded = json.loads(raw)
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("trajectory_json must contain a non-empty JSON list")
    points: list[dict[str, Any]] = []
    for index, source in enumerate(decoded):
        if not isinstance(source, dict):
            raise ValueError(f"trajectory point {index} is not an object")
        point = deepcopy(source)
        try:
            point["t_since_ms"] = float(point["t_since_ms"])
            for axis in XYZ:
                point[axis] = float(point[axis])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"trajectory point {index} lacks finite t/x/y/z values") from exc
        if not all(math.isfinite(point[key]) for key in ("t_since_ms", *XYZ)):
            raise ValueError(f"trajectory point {index} contains non-finite t/x/y/z values")
        point["source_frame"] = point.get("source_frame", point.get("frame", index))
        point["is_original_sample"] = True
        points.append(point)
    points.sort(key=lambda p: p["t_since_ms"])
    times = [p["t_since_ms"] for p in points]
    if len(set(times)) != len(times):
        raise ValueError("trajectory timestamps must be unique")
    return points


def align_trajectory_to_por(
    league_points: Sequence[Mapping[str, Any]], mocap_por: Sequence[float]
) -> list[dict[str, Any]]:
    """Translate every League point by one identical PoR offset."""
    if not league_points:
        raise ValueError("league_points must not be empty")
    target = np.asarray(mocap_por, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("mocap_por must contain three finite coordinates")
    start = np.asarray([league_points[0][axis] for axis in XYZ], dtype=float)
    offset = target - start
    aligned: list[dict[str, Any]] = []
    for source in league_points:
        point = deepcopy(dict(source))
        translated = np.asarray([source[axis] for axis in XYZ], dtype=float) + offset
        point.update(zip(XYZ, translated.tolist()))
        point["is_original_sample"] = bool(source.get("is_original_sample", True))
        aligned.append(point)
    return aligned


def align_bounce_trajectory_to_por(
    points: Sequence[Mapping[str, Any]], mocap_por: Sequence[float]
) -> list[dict[str, Any]]:
    """Align x/y constantly while fading the initial z correction to contact."""
    if not points:
        raise ValueError("points must not be empty")
    bounce = next((p for p in points if p.get("event") == "bounce"), None)
    if bounce is None:
        raise ValueError("bounce-aware alignment requires a synthetic bounce point")
    target = np.asarray(mocap_por, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("mocap_por must contain three finite coordinates")
    start = np.asarray([points[0][axis] for axis in XYZ], dtype=float)
    dx, dy, dz = target - start
    start_time = float(points[0]["t_since_ms"])
    bounce_time = float(bounce["t_since_ms"])
    if bounce_time <= start_time:
        raise ValueError("bounce time must be after the PoR time")

    aligned: list[dict[str, Any]] = []
    for source in points:
        point = deepcopy(dict(source))
        time_ms = float(source["t_since_ms"])
        if time_ms < bounce_time:
            s = min(1.0, max(0.0, (time_ms - start_time) / (bounce_time - start_time)))
            smootherstep = 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3
            z_correction = float(dz) * (1.0 - smootherstep)
        else:
            z_correction = 0.0
        point.update({
            "x": float(source["x"]) + float(dx),
            "y": float(source["y"]) + float(dy),
            "z": float(source["z"]) + z_correction,
        })
        aligned.append(point)
    return aligned


def detect_bounce_window(
    points: Sequence[Mapping[str, Any]], *, min_vertical_change_m: float = 0.03,
    samples_per_side: int = 2,
) -> BounceWindow | None:
    """Find a local z minimum and return non-overlapping fitting samples.

    The minimum itself only needs to be lower than its immediate neighbours.
    The minimum vertical change is evaluated over the complete fitting window:
    at 20 Hz the sample immediately before contact can already be close to the
    ground, so requiring a large change in that single 50 ms interval misses
    otherwise unambiguous bounces.
    """
    if samples_per_side < 2:
        raise ValueError("samples_per_side must be at least 2")
    if len(points) < 4:
        return None
    candidates: list[tuple[float, int]] = []
    for i in range(1, len(points) - 1):
        prev_z, z, next_z = (float(points[j]["z"]) for j in (i - 1, i, i + 1))
        if not (prev_z > z < next_z):
            continue
        start = max(0, i - samples_per_side)
        end = min(len(points) - 1, i + samples_per_side)
        descent = max(float(points[j]["z"]) for j in range(start, i)) - z
        ascent = max(float(points[j]["z"]) for j in range(i + 1, end + 1)) - z
        if descent >= min_vertical_change_m and ascent >= min_vertical_change_m:
            candidates.append((z, i))
    if not candidates:
        return None
    _, index = min(candidates)
    start = max(0, index - samples_per_side)
    end = min(len(points) - 1, index + samples_per_side)
    if index - start < 2 or end - index < 2:
        return None
    return BounceWindow(index, start, end)


def estimate_bounce_time(
    points: Sequence[Mapping[str, Any]], window: BounceWindow, *, ground_z: float = 0.095,
    gravity_m_s2: float = 9.81, max_vertical_fit_rmse_m: float = 0.20,
    max_reconstruction_rmse_m: float = 0.20, max_reconstruction_error_m: float = 0.35,
) -> BounceEvent:
    """Fit two ballistic z legs with a shared, bounded contact time."""
    incoming = points[window.start_idx:window.minimum_idx + 1]
    outgoing = points[window.minimum_idx + 1:window.end_idx + 1]
    if len(incoming) < 2 or len(outgoing) < 2:
        raise ValueError("bounce fit needs at least two samples on each side")
    t_in = np.asarray([p["t_since_ms"] for p in incoming], dtype=float) / 1000.0
    t_out = np.asarray([p["t_since_ms"] for p in outgoing], dtype=float) / 1000.0
    z_in = np.asarray([p["z"] for p in incoming], dtype=float)
    z_out = np.asarray([p["z"] for p in outgoing], dtype=float)
    lower_tb = float(points[window.minimum_idx]["t_since_ms"]) / 1000.0
    upper_tb = float(points[window.minimum_idx + 1]["t_since_ms"]) / 1000.0
    if not upper_tb > lower_tb:
        raise ValueError("invalid bounce-time bounds")

    def model(times: np.ndarray, tb: float, vz: float) -> np.ndarray:
        dt = times - tb
        return ground_z + vz * dt - 0.5 * gravity_m_s2 * dt * dt

    def residual(parameters: np.ndarray) -> np.ndarray:
        tb, vz_in, vz_out = parameters
        return np.concatenate((model(t_in, tb, vz_in) - z_in,
                               model(t_out, tb, vz_out) - z_out))

    slope_in = float(np.polyfit(t_in, z_in, 1)[0])
    slope_out = float(np.polyfit(t_out, z_out, 1)[0])
    initial = np.asarray([(lower_tb + upper_tb) / 2.0, min(slope_in, -0.1), max(slope_out, 0.1)])
    epsilon = 1e-6
    fit = least_squares(residual, initial,
                        bounds=([lower_tb, -100.0, epsilon], [upper_tb, -epsilon, 100.0]))
    tb, fitted_vz_in, fitted_vz_out = map(float, fit.x)
    fit_rmse = float(np.sqrt(np.mean(np.square(residual(fit.x)))))

    start, end = points[window.start_idx], points[window.end_idx]
    reliable_pre = points[:window.start_idx + 1]
    reliable_post = points[window.end_idx:]
    if len(reliable_pre) >= 2:
        pre_t = np.asarray([p["t_since_ms"] for p in reliable_pre], dtype=float) / 1000.0
        boundary_vz_in = float(PchipInterpolator(pre_t, [p["z"] for p in reliable_pre]).derivative()(pre_t[-1]))
    else:
        boundary_vz_in = float(np.polyfit(t_in, z_in, 1)[0])
    if len(reliable_post) >= 2:
        post_t = np.asarray([p["t_since_ms"] for p in reliable_post], dtype=float) / 1000.0
        boundary_vz_out = float(PchipInterpolator(post_t, [p["z"] for p in reliable_post]).derivative()(post_t[0]))
    else:
        boundary_vz_out = float(np.polyfit(t_out, z_out, 1)[0])

    tb_ms = tb * 1000.0
    def hermite(t_ms: float, t0_ms: float, t1_ms: float, z0: float, z1: float,
                v0: float, v1: float) -> float:
        duration = (t1_ms - t0_ms) / 1000.0
        u = (t_ms - t0_ms) / (t1_ms - t0_ms)
        return float((2*u**3 - 3*u**2 + 1) * z0 + (u**3 - 2*u**2 + u) * duration * v0
                     + (-2*u**3 + 3*u**2) * z1 + (u**3 - u**2) * duration * v1)

    errors: list[float] = []
    for point in points[window.start_idx + 1:window.end_idx]:
        time_ms = float(point["t_since_ms"])
        predicted = (hermite(time_ms, float(start["t_since_ms"]), tb_ms, float(start["z"]),
                             ground_z, boundary_vz_in, fitted_vz_in)
                     if time_ms <= tb_ms else
                     hermite(time_ms, tb_ms, float(end["t_since_ms"]), ground_z, float(end["z"]),
                             fitted_vz_out, boundary_vz_out))
        errors.append(predicted - float(point["z"]))
    reconstruction_rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
    reconstruction_max_error = float(max(map(abs, errors), default=0.0))
    valid = bool(fit.success and lower_tb < tb < upper_tb and fitted_vz_in < 0 < fitted_vz_out
                 and fit_rmse <= max_vertical_fit_rmse_m
                 and reconstruction_rmse <= max_reconstruction_rmse_m
                 and reconstruction_max_error <= max_reconstruction_error_m)

    measured = [p for p in points if p.get("is_original_sample", True)]
    times_ms = np.asarray([p["t_since_ms"] for p in measured], dtype=float)
    x_interp = PchipInterpolator(times_ms, [p["x"] for p in measured])
    y_interp = PchipInterpolator(times_ms, [p["y"] for p in measured])
    return BounceEvent(
        time_ms=tb_ms, x=float(x_interp(tb_ms)), y=float(y_interp(tb_ms)),
        z=float(ground_z), vertical_fit_rmse=fit_rmse,
        vz_in_m_s=fitted_vz_in, vz_out_m_s=fitted_vz_out,
        fit_valid=valid, boundary_start_ms=float(start["t_since_ms"]),
        boundary_end_ms=float(end["t_since_ms"]), boundary_vz_in_m_s=boundary_vz_in,
        boundary_vz_out_m_s=boundary_vz_out, reconstruction_rmse_m=reconstruction_rmse,
        reconstruction_max_error_m=reconstruction_max_error,
    )


def insert_bounce_event(points: Sequence[Mapping[str, Any]], event: BounceEvent) -> list[dict[str, Any]]:
    contact = {
        "source_frame": None, "t_since_ms": event.time_ms,
        "x": event.x, "y": event.y, "z": event.z,
        "is_original_sample": False, "event": "bounce",
        "bounce_vertical_fit_rmse": event.vertical_fit_rmse,
        "bounce_vz_in_m_s": event.vz_in_m_s, "bounce_vz_out_m_s": event.vz_out_m_s,
        "bounce_fit_valid": event.fit_valid,
        "bounce_boundary_start_ms": event.boundary_start_ms,
        "bounce_boundary_end_ms": event.boundary_end_ms,
        "bounce_boundary_vz_in_m_s": event.boundary_vz_in_m_s,
        "bounce_boundary_vz_out_m_s": event.boundary_vz_out_m_s,
        "bounce_reconstruction_rmse_m": event.reconstruction_rmse_m,
        "bounce_reconstruction_max_error_m": event.reconstruction_max_error_m,
    }
    result = [deepcopy(dict(point)) for point in points]
    result.append(contact)
    result.sort(key=lambda point: float(point["t_since_ms"]))
    return result


def _target_times(start: float, end: float, target_hz: float) -> np.ndarray:
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be positive and finite")
    step_ms = 1000.0 / target_hz
    count = int(math.floor((end - start) / step_ms + 1e-10))
    times = start + np.arange(count + 1, dtype=float) * step_ms
    if not np.isclose(times[-1], end, rtol=0.0, atol=1e-9):
        times = np.append(times, end)
    else:
        times[-1] = end
    return times


def _interpolate_segment(points: Sequence[Mapping[str, Any]], times: np.ndarray) -> np.ndarray:
    knots = np.asarray([p["t_since_ms"] for p in points], dtype=float)
    values = np.asarray([[p[axis] for axis in XYZ] for p in points], dtype=float)
    if len(points) == 1:
        return np.repeat(values, len(times), axis=0)
    return np.column_stack([PchipInterpolator(knots, values[:, axis])(times) for axis in range(3)])


def _evaluate_trajectory(points: Sequence[Mapping[str, Any]], times: np.ndarray) -> np.ndarray:
    """Evaluate continuous x/y and bounce-aware z at arbitrary timestamps."""
    measured = [p for p in points if p.get("is_original_sample", False)]
    knots = np.asarray([p["t_since_ms"] for p in measured], dtype=float)
    xy = np.column_stack([
        PchipInterpolator(knots, [p[axis] for p in measured])(times) for axis in ("x", "y")
    ])
    z = np.asarray(PchipInterpolator(knots, [p["z"] for p in measured])(times), dtype=float)
    event = next((p for p in points if p.get("event") == "bounce"), None)
    if event is not None:
        tb = float(event["t_since_ms"])
        boundary_start = float(event["bounce_boundary_start_ms"])
        boundary_end = float(event["bounce_boundary_end_ms"])
        start_point = next(p for p in measured if math.isclose(float(p["t_since_ms"]), boundary_start))
        end_point = next(p for p in measured if math.isclose(float(p["t_since_ms"]), boundary_end))
        reliable_pre = [p for p in measured if float(p["t_since_ms"]) <= boundary_start]
        reliable_post = [p for p in measured if float(p["t_since_ms"]) >= boundary_end]
        before_mask = times <= boundary_start
        after_mask = times >= boundary_end
        if len(reliable_pre) >= 2:
            z[before_mask] = PchipInterpolator(
                [p["t_since_ms"] for p in reliable_pre], [p["z"] for p in reliable_pre]
            )(times[before_mask])
        if len(reliable_post) >= 2:
            z[after_mask] = PchipInterpolator(
                [p["t_since_ms"] for p in reliable_post], [p["z"] for p in reliable_post]
            )(times[after_mask])
        pre_mask = (times >= boundary_start) & (times <= tb)
        post_mask = (times > tb) & (times <= boundary_end)
        ground = float(event["z"])

        def hermite_array(sample_times: np.ndarray, t0: float, t1: float, z0: float, z1: float,
                          v0: float, v1: float) -> np.ndarray:
            duration = (t1 - t0) / 1000.0
            u = (sample_times - t0) / (t1 - t0)
            return ((2*u**3 - 3*u**2 + 1) * z0 + (u**3 - 2*u**2 + u) * duration * v0
                    + (-2*u**3 + 3*u**2) * z1 + (u**3 - u**2) * duration * v1)

        z[pre_mask] = hermite_array(
            times[pre_mask], boundary_start, tb, float(start_point["z"]), ground,
            float(event["bounce_boundary_vz_in_m_s"]), float(event["bounce_vz_in_m_s"]),
        )
        z[post_mask] = hermite_array(
            times[post_mask], tb, boundary_end, ground, float(end_point["z"]),
            float(event["bounce_vz_out_m_s"]), float(event["bounce_boundary_vz_out_m_s"]),
        )
    return np.column_stack((xy, z))


def upsample_trajectory(
    points: Sequence[Mapping[str, Any]], target_hz: float = 300.0
) -> list[dict[str, Any]]:
    """Evaluate PCHIP at a dense time grid, split at a synthetic bounce."""
    if not points:
        raise ValueError("points must not be empty")
    start, end = float(points[0]["t_since_ms"]), float(points[-1]["t_since_ms"])
    if end < start:
        raise ValueError("trajectory timestamps must increase")
    times = _target_times(start, end, target_hz)
    bounce_points = [p for p in points if p.get("event") == "bounce"]
    if len(bounce_points) > 1:
        raise ValueError("only one bounce is supported")
    if bounce_points:
        bounce_time = float(bounce_points[0]["t_since_ms"])
        # Retain the event and both transition boundaries even when they fall
        # between regular 300 Hz samples.
        required = (bounce_time, float(bounce_points[0]["bounce_boundary_start_ms"]),
                    float(bounce_points[0]["bounce_boundary_end_ms"]))
        for required_time in required:
            if not np.any(np.isclose(times, required_time, rtol=0.0, atol=1e-9)):
                times = np.append(times, required_time)
        times = np.sort(times)
    xyz = _evaluate_trajectory(points, times)

    original_by_time = {
        float(p["t_since_ms"]): p for p in points if p.get("is_original_sample", False)
    }
    result: list[dict[str, Any]] = []
    for frame, (time_ms, position) in enumerate(zip(times, xyz)):
        original = next((p for t, p in original_by_time.items()
                         if math.isclose(t, float(time_ms), abs_tol=1e-9)), None)
        point: dict[str, Any] = {
            "frame": frame, "t_since_ms": float(time_ms),
            "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
            "is_original_sample": original is not None,
        }
        if original is not None:
            point["source_frame"] = original.get("source_frame")
        if bounce_points and math.isclose(float(time_ms), float(bounce_points[0]["t_since_ms"]), abs_tol=1e-9):
            # Avoid exposing tiny interpolation round-off at the contact knot.
            point.update({axis: float(bounce_points[0][axis]) for axis in XYZ})
            point["event"] = "bounce"
            for key in ("bounce_vertical_fit_rmse", "bounce_vz_in_m_s", "bounce_vz_out_m_s",
                        "bounce_fit_valid", "bounce_boundary_start_ms", "bounce_boundary_end_ms",
                        "bounce_boundary_vz_in_m_s", "bounce_boundary_vz_out_m_s",
                        "bounce_reconstruction_rmse_m", "bounce_reconstruction_max_error_m"):
                point[key] = bounce_points[0][key]
        result.append(point)
    return result


def recompute_kinematics(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive velocity, acceleration, speed, direction, and elevation from positions."""
    result = [deepcopy(dict(point)) for point in points]
    if not result:
        return result
    if len(result) == 1:
        velocity = np.zeros((1, 3), dtype=float)
        acceleration = np.zeros((1, 3), dtype=float)
    else:
        seconds = np.asarray([p["t_since_ms"] for p in result], dtype=float) / 1000.0
        xyz = np.asarray([[p[axis] for axis in XYZ] for p in result], dtype=float)
        edge_order = 2 if len(result) >= 3 else 1
        velocity = np.column_stack([np.gradient(xyz[:, i], seconds, edge_order=edge_order) for i in range(3)])
        bounce = next((i for i, p in enumerate(result) if p.get("event") == "bounce"), None)
        if bounce is not None and 0 < bounce < len(result) - 1:
            # Never central-difference across the impact discontinuity.
            for indices in (np.arange(0, bounce + 1), np.arange(bounce, len(result))):
                local_edge = 2 if len(indices) >= 3 else 1
                local_vz = np.gradient(xyz[indices, 2], seconds[indices], edge_order=local_edge)
                velocity[indices, 2] = local_vz
            velocity[bounce, 2] = float(result[bounce]["bounce_vz_out_m_s"])
        acceleration = np.column_stack([np.gradient(velocity[:, i], seconds, edge_order=edge_order) for i in range(3)])
        if bounce is not None and 0 < bounce < len(result) - 1:
            pre = np.arange(0, bounce + 1)
            post = np.arange(bounce, len(result))
            pre_vz = velocity[pre, 2].copy()
            pre_vz[-1] = float(result[bounce]["bounce_vz_in_m_s"])
            post_vz = velocity[post, 2].copy()
            post_vz[0] = float(result[bounce]["bounce_vz_out_m_s"])
            acceleration[pre, 2] = np.gradient(
                pre_vz, seconds[pre], edge_order=2 if len(pre) >= 3 else 1
            )
            acceleration[post, 2] = np.gradient(
                post_vz, seconds[post], edge_order=2 if len(post) >= 3 else 1
            )
    for point, vel, acc in zip(result, velocity, acceleration):
        speed = float(np.linalg.norm(vel))
        horizontal = float(np.hypot(vel[0], vel[1]))
        point.update({
            "vx": float(vel[0]), "vy": float(vel[1]), "vz": float(vel[2]), "v": speed,
            "ax": float(acc[0]), "ay": float(acc[1]), "az": float(acc[2]),
            "a": float(np.linalg.norm(acc)),
            "dir": float(np.degrees(np.arctan2(vel[1], vel[0]))),
            "vert_angle": float(np.degrees(np.arctan2(vel[2], horizontal))),
        })
        if point.get("event") == "bounce":
            point["vz_before_m_s"] = float(point["bounce_vz_in_m_s"])
            point["vz_after_m_s"] = float(point["bounce_vz_out_m_s"])
    return result


def validate_bounce_reconstruction(points: Sequence[Mapping[str, Any]], *, atol: float = 1e-6) -> None:
    """Validate the hard z cusp and continuous, unmodified horizontal path."""
    bounce = next((i for i, p in enumerate(points) if p.get("event") == "bounce"), None)
    if bounce is None:
        return
    if not 0 < bounce < len(points) - 1:
        raise AssertionError("bounce must have samples on both sides")
    point = points[bounce]
    assert float(point["bounce_vz_in_m_s"]) < 0 < float(point["bounce_vz_out_m_s"])
    assert float(point["z"]) < float(points[bounce - 1]["z"])
    assert float(point["z"]) < float(points[bounce + 1]["z"])
    for axis in ("x", "y"):
        assert math.isfinite(float(point[axis]))
    t0, tb, t1 = (float(points[i]["t_since_ms"]) / 1000.0 for i in (bounce - 1, bounce, bounce + 1))
    left = np.asarray([(point[a] - points[bounce - 1][a]) / (tb - t0) for a in ("x", "y")])
    right = np.asarray([(points[bounce + 1][a] - point[a]) / (t1 - tb) for a in ("x", "y")])
    np.testing.assert_allclose(left, right, rtol=0.15, atol=max(atol, 0.15))


def validate_reconstruction(
    original: Sequence[Mapping[str, Any]], aligned: Sequence[Mapping[str, Any]],
    control_points: Sequence[Mapping[str, Any]], mocap_por: Sequence[float], *, atol: float = 1e-6,
) -> None:
    """Raise AssertionError if a reconstruction geometry/timing invariant is broken."""
    original_xyz = np.asarray([[p[a] for a in XYZ] for p in original], dtype=float)
    aligned_xyz = np.asarray([[p[a] for a in XYZ] for p in aligned], dtype=float)
    np.testing.assert_allclose(aligned_xyz[0], mocap_por, atol=atol, rtol=0)
    offsets = aligned_xyz - original_xyz
    np.testing.assert_allclose(offsets, np.repeat(offsets[:1], len(offsets), axis=0), atol=atol, rtol=0)
    np.testing.assert_allclose(aligned_xyz[:, None, :] - aligned_xyz[None, :, :],
                               original_xyz[:, None, :] - original_xyz[None, :, :], atol=atol, rtol=0)
    event = next((p for p in control_points if p.get("event") == "bounce"), None)
    exact_indices = list(range(len(aligned)))
    if event is not None:
        start = float(event["bounce_boundary_start_ms"])
        end = float(event["bounce_boundary_end_ms"])
        exact_indices = [i for i, p in enumerate(aligned)
                         if float(p["t_since_ms"]) <= start or float(p["t_since_ms"]) >= end]
    exact_times = [aligned[i]["t_since_ms"] for i in exact_indices]
    interpolated = upsample_at_times(control_points, exact_times)
    np.testing.assert_allclose([[p[a] for a in XYZ] for p in interpolated],
                               aligned_xyz[exact_indices], atol=atol, rtol=0)
    assert math.isclose(float(original[0]["t_since_ms"]), float(control_points[0]["t_since_ms"]), abs_tol=atol)
    assert math.isclose(float(original[-1]["t_since_ms"]), float(control_points[-1]["t_since_ms"]), abs_tol=atol)


def validate_bounce_alignment(
    original: Sequence[Mapping[str, Any]], aligned: Sequence[Mapping[str, Any]],
    mocap_por: Sequence[float], ground_z: float, *, atol: float = 1e-6,
) -> None:
    """Validate PoR retargeting without moving the floor or post-bounce path."""
    if len(original) != len(aligned):
        raise AssertionError("bounce alignment must preserve the number of samples")
    original_times = np.asarray([p["t_since_ms"] for p in original], dtype=float)
    aligned_times = np.asarray([p["t_since_ms"] for p in aligned], dtype=float)
    np.testing.assert_allclose(aligned_times, original_times, atol=atol, rtol=0)
    np.testing.assert_allclose([aligned[0][a] for a in XYZ], mocap_por, atol=atol, rtol=0)

    original_xyz = np.asarray([[p[a] for a in XYZ] for p in original], dtype=float)
    aligned_xyz = np.asarray([[p[a] for a in XYZ] for p in aligned], dtype=float)
    offsets = aligned_xyz - original_xyz
    np.testing.assert_allclose(offsets[:, :2], np.repeat(offsets[:1, :2], len(offsets), axis=0),
                               atol=atol, rtol=0)
    bounce_index = next((i for i, p in enumerate(original) if p.get("event") == "bounce"), None)
    if bounce_index is None:
        raise AssertionError("bounce-aware alignment lost the bounce event")
    assert math.isclose(float(aligned[bounce_index]["z"]), ground_z, abs_tol=atol)
    assert math.isclose(float(offsets[0, 2]), float(mocap_por[2]) - original_xyz[0, 2], abs_tol=atol)
    assert math.isclose(float(offsets[bounce_index, 2]), 0.0, abs_tol=atol)
    np.testing.assert_allclose(offsets[bounce_index:, 2], 0.0, atol=atol, rtol=0)


def upsample_at_times(points: Sequence[Mapping[str, Any]], times_ms: Sequence[float]) -> list[dict[str, float]]:
    """Evaluate the same piecewise interpolant at caller-provided timestamps."""
    times = np.asarray(times_ms, dtype=float)
    xyz = _evaluate_trajectory(points, times)
    return [{"t_since_ms": float(t), **dict(zip(XYZ, map(float, pos)))} for t, pos in zip(times, xyz)]


def reconstruct_league_continuation(
    league_throw: Mapping[str, Any], mocap_throw: Mapping[str, Any], target_hz: float = 300.0,
    *, ground_z: float = 0.095,
) -> list[dict[str, Any]]:
    original_relative = parse_trajectory_json(league_throw)
    league_por = np.asarray([float(league_throw[f"por_{axis}_m"]) for axis in XYZ])
    original_absolute: list[dict[str, Any]] = []
    for source in original_relative:
        point = deepcopy(source)
        point.update({axis: float(source[axis]) + float(league_por[i])
                      for i, axis in enumerate(XYZ)})
        original_absolute.append(point)
    mocap_por = tuple(float(mocap_throw[f"por_{axis}_m"]) for axis in XYZ)
    window = detect_bounce_window(original_absolute)
    event = estimate_bounce_time(original_absolute, window, ground_z=ground_z) if window else None
    controls_absolute = insert_bounce_event(original_absolute, event) if event else original_absolute
    if event is not None:
        upsampled_absolute = upsample_trajectory(controls_absolute, target_hz)
        upsampled = align_bounce_trajectory_to_por(upsampled_absolute, mocap_por)
        validate_bounce_alignment(upsampled_absolute, upsampled, mocap_por, ground_z)
    else:
        aligned_original = align_trajectory_to_por(original_absolute, mocap_por)
        aligned_controls = align_trajectory_to_por(controls_absolute, mocap_por)
        validate_reconstruction(original_absolute, aligned_original, aligned_controls, mocap_por)
        upsampled = upsample_trajectory(aligned_controls, target_hz)
    validate_bounce_reconstruction(upsampled)
    reconstructed = recompute_kinematics(upsampled)
    assert math.isclose(reconstructed[0]["t_since_ms"], original_relative[0]["t_since_ms"], abs_tol=1e-9)
    assert math.isclose(reconstructed[-1]["t_since_ms"], original_relative[-1]["t_since_ms"], abs_tol=1e-9)
    return reconstructed


def reconstruct_matches(
    matches_csv: str | Path, raw_league_csv: str | Path, raw_mocap_csv: str | Path,
    output_csv: str | Path, *, target_hz: float = 300.0, rank: int | None = None,
    ground_z: float = 0.095,
    weight_run_id: int | None = None, weight_set: Mapping[str, float] | None = None,
    skip_invalid: bool = False, errors_csv: str | Path | None = None,
) -> int:
    matches, _ = _read_csv(Path(matches_csv))
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mocap_throw_id", "match_rank", "league_throw_id", "weight_run_id", "weights_json",
                  "target_sampling_rate_hz",
                  "translation_x_m", "translation_y_m", "translation_z_m", "z_alignment_mode",
                  "bounce_detected", "bounce_time_ms", "bounce_x_m", "bounce_y_m", "bounce_z_m",
                  "bounce_vertical_fit_rmse", "bounce_vz_in_m_s", "bounce_vz_out_m_s",
                  "bounce_reconstruction_rmse_m", "bounce_reconstruction_max_error_m",
                  "bounce_fit_valid", "trajectory_json"]
    error_fieldnames = [
        "mocap_throw_id", "match_rank", "league_throw_id", "error_type", "error",
    ]
    errors: list[dict[str, Any]] = []
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            match_rank = int(float(match.get("rank", 1)))
            if rank is not None and match_rank != rank:
                continue
            try:
                league = load_raw_throw(raw_league_csv, match["league_throw_id"])
                mocap = load_raw_throw(raw_mocap_csv, match["mocap_throw_id"])
                league_por = np.asarray([float(league[f"por_{a}_m"]) for a in XYZ])
                por = np.asarray([float(mocap[f"por_{a}_m"]) for a in XYZ])
                offset = por - league_por
                reconstructed = reconstruct_league_continuation(
                    league, mocap, target_hz, ground_z=ground_z,
                )
            except Exception as error:
                if not skip_invalid:
                    raise
                detail = str(error).strip() or "reconstruction validation failed"
                errors.append({
                    "mocap_throw_id": match.get("mocap_throw_id", ""),
                    "match_rank": match_rank,
                    "league_throw_id": match.get("league_throw_id", ""),
                    "error_type": type(error).__name__,
                    "error": detail,
                })
                continue
            bounce_point = next((p for p in reconstructed if p.get("event") == "bounce"), None)
            writer.writerow({
                "mocap_throw_id": match["mocap_throw_id"],
                "match_rank": match_rank,
                "league_throw_id": match["league_throw_id"],
                "weight_run_id": "" if weight_run_id is None else weight_run_id,
                "weights_json": "" if weight_set is None else json.dumps(weight_set, sort_keys=True),
                "target_sampling_rate_hz": target_hz,
                "translation_x_m": offset[0], "translation_y_m": offset[1], "translation_z_m": offset[2],
                "z_alignment_mode": "fade_to_ground" if bounce_point is not None else "constant",
                "bounce_detected": bounce_point is not None,
                "bounce_time_ms": "" if bounce_point is None else bounce_point["t_since_ms"],
                "bounce_x_m": "" if bounce_point is None else bounce_point["x"],
                "bounce_y_m": "" if bounce_point is None else bounce_point["y"],
                "bounce_z_m": "" if bounce_point is None else bounce_point["z"],
                "bounce_vertical_fit_rmse": "" if bounce_point is None else bounce_point["bounce_vertical_fit_rmse"],
                "bounce_vz_in_m_s": "" if bounce_point is None else bounce_point["bounce_vz_in_m_s"],
                "bounce_vz_out_m_s": "" if bounce_point is None else bounce_point["bounce_vz_out_m_s"],
                "bounce_reconstruction_rmse_m": "" if bounce_point is None else bounce_point["bounce_reconstruction_rmse_m"],
                "bounce_reconstruction_max_error_m": "" if bounce_point is None else bounce_point["bounce_reconstruction_max_error_m"],
                "bounce_fit_valid": "" if bounce_point is None else bounce_point["bounce_fit_valid"],
                "trajectory_json": json.dumps(reconstructed, separators=(",", ":"), allow_nan=False),
            })
            count += 1
    if skip_invalid:
        errors_path = (
            Path(errors_csv) if errors_csv is not None
            else output_path.with_name(f"{output_path.stem}_errors{output_path.suffix}")
        )
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=error_fieldnames)
            writer.writeheader()
            writer.writerows(errors)
    return count


def resolve_random_weight_run(search_dir: str | Path, run_id: int) -> tuple[Path, dict[str, float]]:
    """Resolve one random-search run's matches and exact saved weights."""
    if run_id < 1:
        raise ValueError("random run ID must be at least 1")
    run_dir = Path(search_dir) / f"run_{run_id:04d}"
    matches = run_dir / "weighted_knn_matches.csv"
    weights_path = run_dir / "weights.json"
    if not matches.is_file():
        raise FileNotFoundError(f"Random-search match file not found: {matches}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Random-search weights file not found: {weights_path}")
    parsed = json.loads(weights_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a weight dictionary in {weights_path}")
    return matches, {str(key): float(value) for key, value in parsed.items()}


def output_path_for_run(path: str | Path, run_id: int) -> Path:
    """Add a stable run_XXXX suffix without duplicating an existing suffix."""
    output = Path(path)
    suffix = f"_run_{run_id:04d}"
    stem = output.stem if output.stem.endswith(suffix) else output.stem + suffix
    return output.with_name(stem + output.suffix)


def select_random_weight_run(
    search_dir: str | Path, weight_groups: Sequence[str], *, top_runs: int = 20,
) -> tuple[int, float, float]:
    """Select the most jointly important requested weights among top runs.

    Top runs are determined by ascending balanced mean top-1 distance. Within
    that shortlist, the geometric mean rewards configurations where every
    requested group has a high weight. Returns run ID, selection value, and
    balanced distance.
    """
    if top_runs < 1:
        raise ValueError("top_runs must be at least 1")
    groups = list(dict.fromkeys(weight_groups))
    if not groups:
        raise ValueError("at least one weight group must be selected")
    unknown = set(groups) - set(WEIGHT_GROUP_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown weight groups {sorted(unknown)}; available: {list(WEIGHT_GROUP_NAMES)}"
        )
    summary_path = Path(search_dir) / "random_search_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Random-search summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {"run", "balanced_mean_top1_distance", *groups}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Random-search summary is missing columns: {sorted(missing)}")
    summary["balanced_mean_top1_distance"] = pd.to_numeric(
        summary["balanced_mean_top1_distance"], errors="coerce"
    )
    for group in groups:
        summary[group] = pd.to_numeric(summary[group], errors="coerce")
    valid = summary.dropna(subset=["run", "balanced_mean_top1_distance", *groups]).copy()
    valid = valid[np.isfinite(valid["balanced_mean_top1_distance"])]
    valid = valid[(valid[groups] >= 0).all(axis=1)]
    if valid.empty:
        raise ValueError("No valid random-search rows remain for automatic run selection")
    shortlist = valid.sort_values(
        ["balanced_mean_top1_distance", "run"], kind="stable"
    ).head(top_runs).copy()
    values = shortlist[groups].to_numpy(dtype=float)
    shortlist["selected_weight_geometric_mean"] = np.prod(values, axis=1) ** (1.0 / len(groups))
    chosen = shortlist.sort_values(
        ["selected_weight_geometric_mean", "balanced_mean_top1_distance", "run"],
        ascending=[False, True, True], kind="stable",
    ).iloc[0]
    return (int(chosen["run"]), float(chosen["selected_weight_geometric_mean"]),
            float(chosen["balanced_mean_top1_distance"]))


def main() -> int:
    base = Path(__file__).resolve().parents[1] / "out"
    throw_features = base / "throw_features"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=base / "weighted_knn_matches.csv")
    parser.add_argument(
        "--random-search-dir", type=Path,
        help="weighted_knn_random_* directory containing saved run_XXXX folders",
    )
    run_selection = parser.add_mutually_exclusive_group()
    run_selection.add_argument(
        "--random-run-id", type=int,
        help="use weighted_knn_matches.csv and weights.json from this random-search run",
    )
    run_selection.add_argument(
        "--select-weight-groups", nargs="+", metavar="GROUP",
        help="among the top runs, select the run with the highest joint importance of these weights",
    )
    parser.add_argument(
        "--top-runs", type=int, default=20, metavar="N",
        help="number of lowest-distance runs considered by --select-weight-groups (default: 20)",
    )
    parser.add_argument("--raw-league", type=Path, default=throw_features / "raw_league.csv")
    parser.add_argument("--raw-mocap", type=Path, default=throw_features / "raw_mocap.csv")
    parser.add_argument("--output", type=Path, default=base / "reconstructed_matched_trajectories.csv")
    parser.add_argument("--target-hz", type=float, default=300.0)
    parser.add_argument(
        "--rank", type=int,
        help="reconstruct only this neighbor rank (default: reconstruct every match row)",
    )
    parser.add_argument(
        "--skip-invalid", action="store_true",
        help=("continue after invalid match reconstructions and write them to "
              "<output stem>_errors.csv"),
    )
    parser.add_argument(
        "--errors-output", type=Path,
        help="error-report CSV used with --skip-invalid (default: beside --output)",
    )
    parser.add_argument("--ground-z", type=float, default=0.095,
                        help="ball-centre height at ground contact in metres (default: 0.095)")
    args = parser.parse_args()
    wants_random_run = args.random_run_id is not None or args.select_weight_groups is not None
    if wants_random_run and args.random_search_dir is None:
        parser.error("--random-search-dir is required with random run selection")
    if args.random_search_dir is not None and not wants_random_run:
        parser.error("--random-search-dir requires --random-run-id or --select-weight-groups")
    if args.top_runs < 1:
        parser.error("--top-runs must be at least 1")
    matches = args.matches
    output = args.output
    weights = None
    if args.select_weight_groups is not None:
        try:
            selected_run, importance, distance = select_random_weight_run(
                args.random_search_dir, args.select_weight_groups, top_runs=args.top_runs,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        args.random_run_id = selected_run
        print(f"Selected run {selected_run:04d} from the {args.top_runs} lowest-distance runs")
        print(f"Weight groups: {', '.join(args.select_weight_groups)}")
        print(f"Joint weight geometric mean: {importance:.8g}")
        print(f"Balanced mean top-1 distance: {distance:.8g}")
    if args.random_run_id is not None:
        matches, weights = resolve_random_weight_run(args.random_search_dir, args.random_run_id)
        output = output_path_for_run(output, args.random_run_id)
        print(f"Using random-search run {args.random_run_id:04d}: {matches}")
        print(f"Weights: {json.dumps(weights, sort_keys=True)}")
    count = reconstruct_matches(
        matches, args.raw_league, args.raw_mocap, output, target_hz=args.target_hz,
        rank=args.rank, ground_z=args.ground_z, weight_run_id=args.random_run_id,
        weight_set=weights, skip_invalid=args.skip_invalid, errors_csv=args.errors_output,
    )
    print(f"Wrote {count} reconstructed trajectories to {output}")
    if args.skip_invalid:
        errors_output = args.errors_output or output.with_name(
            f"{output.stem}_errors{output.suffix}"
        )
        error_rows, _ = _read_csv(errors_output)
        print(f"Skipped {len(error_rows)} invalid matches; details: {errors_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
