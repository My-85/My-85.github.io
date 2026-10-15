"""Maneuver-agnostic differential geometry for sampled 3-D flight paths.

Positions use the repository convention ``[north, east, altitude]`` with
altitude positive upward.  Attitude frames are transported in NED, which is the
right-handed world convention used by the quaternion environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_EPS = 1e-9


def _normalise(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > _EPS:
        return vector / norm
    if fallback is None:
        raise ValueError("cannot normalise a zero vector")
    return _normalise(np.asarray(fallback, dtype=np.float64))


def _smooth_rows(values: np.ndarray, passes: int) -> np.ndarray:
    """Short symmetric filter that preserves first-order endpoint trends.

    Replicating the first/last sample outside an open path silently imposes a
    zero derivative on every smoothing pass.  When this helper is applied to
    tangents, that artificial boundary condition tapers a perfectly constant
    curvature turn toward zero at both ends.  A force-frame compiler then asks
    for a large, nonphysical bank step immediately after take-off and just
    before the terminal row.  Linear extrapolation preserves affine signals
    under the symmetric kernel and leaves the endpoint curvature one-sided,
    as an open path requires.
    """
    result = np.asarray(values, dtype=np.float64).copy()
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    for _ in range(max(0, int(passes))):
        if len(result) == 1:
            padded = np.repeat(result, 5, axis=0)
        else:
            first_delta = result[1] - result[0]
            last_delta = result[-1] - result[-2]
            padded = np.concatenate(
                (
                    np.asarray(
                        [result[0] - 2.0 * first_delta, result[0] - first_delta]
                    ),
                    result,
                    np.asarray(
                        [result[-1] + last_delta, result[-1] + 2.0 * last_delta]
                    ),
                ),
                axis=0,
            )
        result = sum(kernel[index] * padded[index:index + len(result)] for index in range(5))
    return result


def _minimal_rotation(vector: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Parallel-transport ``vector`` through the shortest source→target rotation."""
    source = _normalise(source)
    target = _normalise(target)
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.asarray(vector, dtype=np.float64)
        # An exact tangent reversal is ill-defined.  Rotate by pi around a
        # stable normal axis; well-sampled regular paths should not hit this.
        axis = _normalise(vector - np.dot(vector, source) * source, np.asarray([0.0, 1.0, 0.0]))
        return 2.0 * axis * np.dot(axis, vector) - vector
    axis /= sine
    value = np.asarray(vector, dtype=np.float64)
    # Rodrigues formula, with sin(angle)=sine and cos(angle)=cosine.
    return (
        cosine * value
        + sine * np.cross(axis, value)
        + (1.0 - cosine) * axis * np.dot(axis, value)
    )


@dataclass(frozen=True)
class GeometrySample:
    s_m: float
    position_neu_m: np.ndarray
    tangent_neu: np.ndarray
    tangent_ned: np.ndarray
    curvature_neu_per_m: np.ndarray
    curvature_ned_per_m: np.ndarray
    curvature_per_m: float
    torsion_per_m: float
    transported_down_ned: np.ndarray
    local_radius_m: float


class PathGeometry:
    """Arc-length-resampled geometry and nonsingular transported path frame."""

    def __init__(
        self,
        waypoints: np.ndarray,
        spacing_m: float = 25.0,
        tangent_smoothing_passes: int = 2,
    ) -> None:
        raw = np.asarray(waypoints, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 2:
            raise ValueError("waypoints must have shape [N>=2, 3]")
        if not np.all(np.isfinite(raw)):
            raise ValueError("waypoints contain non-finite values")
        keep = np.concatenate(([True], np.linalg.norm(np.diff(raw, axis=0), axis=1) > 1e-6))
        raw = raw[keep]
        if len(raw) < 2:
            raise ValueError("path has no non-zero-length segment")

        raw_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))))
        self.total_length_m = float(raw_arc[-1])
        # The second-order endpoint derivatives below require at least three
        # samples.  A valid short segment must remain compilable even when its
        # total length is smaller than the requested geometry spacing.
        count = max(3, int(math.ceil(self.total_length_m / max(float(spacing_m), 1.0))) + 1)
        self.arc_m = np.linspace(0.0, self.total_length_m, count)
        self.positions_neu_m = np.column_stack(
            [np.interp(self.arc_m, raw_arc, raw[:, axis]) for axis in range(3)]
        )

        derivative = np.gradient(self.positions_neu_m, self.arc_m, axis=0, edge_order=2)
        derivative = _smooth_rows(derivative, tangent_smoothing_passes)
        tangent_neu = np.asarray([_normalise(row, np.asarray([1.0, 0.0, 0.0])) for row in derivative])
        # Re-normalise after a final light smoothing to suppress polyline vertex impulses.
        tangent_neu = _smooth_rows(tangent_neu, 1)
        self.tangent_neu = np.asarray([_normalise(row) for row in tangent_neu])
        self.curvature_neu_per_m = np.gradient(self.tangent_neu, self.arc_m, axis=0, edge_order=2)
        self.curvature_neu_per_m -= (
            np.sum(self.curvature_neu_per_m * self.tangent_neu, axis=1)[:, None]
            * self.tangent_neu
        )
        self.curvature_per_m = np.linalg.norm(self.curvature_neu_per_m, axis=1)

        third = np.gradient(self.curvature_neu_per_m, self.arc_m, axis=0, edge_order=2)
        cross = np.cross(self.tangent_neu, self.curvature_neu_per_m)
        denominator = np.sum(cross * cross, axis=1)
        numerator = np.einsum("ij,ij->i", cross, third)
        self.torsion_per_m = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        self.torsion_per_m = np.nan_to_num(self.torsion_per_m, nan=0.0, posinf=0.0, neginf=0.0)

        tangent_ned = self.tangent_neu.copy()
        tangent_ned[:, 2] *= -1.0
        self.tangent_ned = tangent_ned
        curvature_ned = self.curvature_neu_per_m.copy()
        curvature_ned[:, 2] *= -1.0
        self.curvature_ned_per_m = curvature_ned
        self.transported_down_ned = self._transport_down(tangent_ned)

    @staticmethod
    def _transport_down(tangents_ned: np.ndarray) -> np.ndarray:
        down_world = np.asarray([0.0, 0.0, 1.0])
        initial = down_world - float(np.dot(down_world, tangents_ned[0])) * tangents_ned[0]
        if np.linalg.norm(initial) < 1e-6:
            east = np.asarray([0.0, 1.0, 0.0])
            initial = east - float(np.dot(east, tangents_ned[0])) * tangents_ned[0]
        frames = np.empty_like(tangents_ned)
        frames[0] = _normalise(initial)
        for index in range(1, len(tangents_ned)):
            transported = _minimal_rotation(frames[index - 1], tangents_ned[index - 1], tangents_ned[index])
            transported -= float(np.dot(transported, tangents_ned[index])) * tangents_ned[index]
            frames[index] = _normalise(transported, frames[index - 1])
        return frames

    def _interp(self, values: np.ndarray, s_m: float) -> np.ndarray:
        s_m = float(np.clip(s_m, 0.0, self.total_length_m))
        if values.ndim == 1:
            return np.asarray(np.interp(s_m, self.arc_m, values))
        return np.asarray([np.interp(s_m, self.arc_m, values[:, axis]) for axis in range(values.shape[1])])

    def sample(self, s_m: float) -> GeometrySample:
        s_m = float(np.clip(s_m, 0.0, self.total_length_m))
        tangent_neu = _normalise(self._interp(self.tangent_neu, s_m))
        tangent_ned = tangent_neu.copy()
        tangent_ned[2] *= -1.0
        curvature_neu = self._interp(self.curvature_neu_per_m, s_m)
        curvature_neu -= float(np.dot(curvature_neu, tangent_neu)) * tangent_neu
        curvature_ned = curvature_neu.copy()
        curvature_ned[2] *= -1.0
        transported_down = self._interp(self.transported_down_ned, s_m)
        transported_down -= float(np.dot(transported_down, tangent_ned)) * tangent_ned
        transported_down = _normalise(transported_down, np.asarray([0.0, 0.0, 1.0]))
        curvature = float(np.linalg.norm(curvature_neu))
        return GeometrySample(
            s_m=s_m,
            position_neu_m=self._interp(self.positions_neu_m, s_m),
            tangent_neu=tangent_neu,
            tangent_ned=tangent_ned,
            curvature_neu_per_m=curvature_neu,
            curvature_ned_per_m=curvature_ned,
            curvature_per_m=curvature,
            torsion_per_m=float(self._interp(self.torsion_per_m, s_m)),
            transported_down_ned=transported_down,
            local_radius_m=(1.0 / curvature if curvature > 1e-8 else float("inf")),
        )

    def project_local(
        self,
        position_neu_m: np.ndarray,
        center_s_m: float,
        backtrack_m: float = 100.0,
        forward_m: float = 2200.0,
    ) -> tuple[float, np.ndarray, float, int]:
        """Project onto a phase-local segment window, safe at crossings/loops."""
        position = np.asarray(position_neu_m, dtype=np.float64)
        low_s = max(0.0, float(center_s_m) - float(backtrack_m))
        high_s = min(self.total_length_m, float(center_s_m) + float(forward_m))
        low = max(0, int(np.searchsorted(self.arc_m, low_s, side="right")) - 1)
        high = min(len(self.arc_m) - 1, int(np.searchsorted(self.arc_m, high_s, side="left")) + 1)
        starts = self.positions_neu_m[low:high]
        ends = self.positions_neu_m[low + 1:high + 1]
        segments = ends - starts
        lengths_sq = np.maximum(np.sum(segments * segments, axis=1), _EPS)
        fraction = np.clip(np.sum((position - starts) * segments, axis=1) / lengths_sq, 0.0, 1.0)
        projections = starts + fraction[:, None] * segments
        distances = np.linalg.norm(position - projections, axis=1)
        local_index = int(np.argmin(distances))
        index = low + local_index
        s_m = float(self.arc_m[index] + fraction[local_index] * (self.arc_m[index + 1] - self.arc_m[index]))
        return s_m, projections[local_index], float(distances[local_index]), index

    def open_path_cross_track_error(
        self,
        position_neu_m: np.ndarray,
        finite_path_cte_m: float,
    ) -> float:
        """Distance to the finite path united with its forward terminal ray."""
        terminal = self.sample(self.total_length_m)
        delta = np.asarray(position_neu_m, dtype=np.float64) - terminal.position_neu_m
        along_m = float(np.dot(delta, terminal.tangent_neu))
        if along_m < 0.0:
            return float(finite_path_cte_m)
        cross = delta - along_m * terminal.tangent_neu
        return min(float(finite_path_cte_m), float(np.linalg.norm(cross)))

    @property
    def finite_radius_p10_m(self) -> float:
        curved = self.curvature_per_m[self.curvature_per_m > 1e-7]
        if curved.size == 0:
            return float("inf")
        return float(np.percentile(1.0 / curved, 10.0))


class PhaseTracker:
    """Monotone local phase estimator with bounded one-step advancement."""

    def __init__(
        self,
        geometry: PathGeometry,
        control_dt_s: float = 0.2,
        forward_window_m: float = 2200.0,
    ) -> None:
        self.geometry = geometry
        self.control_dt_s = float(control_dt_s)
        self.forward_window_m = float(forward_window_m)
        self.s_m = 0.0
        self.segment_index = 0
        self.last_cte_m = 0.0
        self.last_progress_scale = 1.0
        self.capture_lost = False
        self.last_projected_s_m = 0.0

    def reset(self, s_m: float = 0.0) -> None:
        self.s_m = float(np.clip(s_m, 0.0, self.geometry.total_length_m))
        self.segment_index = int(np.searchsorted(self.geometry.arc_m, self.s_m, side="right") - 1)
        self.segment_index = int(np.clip(self.segment_index, 0, len(self.geometry.arc_m) - 2))
        self.last_cte_m = 0.0
        self.last_progress_scale = 1.0
        self.capture_lost = False
        self.last_projected_s_m = self.s_m

    def update(
        self,
        position_neu_m: np.ndarray,
        airspeed_mps: float,
        cross_track_scale_m: float = 0.0,
        minimum_progress_scale: float = 0.0,
        capture_radius_m: float = 0.0,
        capture_release_ratio: float = 0.8,
        capture_release_allowed: bool = True,
    ) -> tuple[float, float]:
        candidate, _, cte, index = self.geometry.project_local(
            position_neu_m,
            self.s_m,
            backtrack_m=max(75.0, 1.5 * float(airspeed_mps) * self.control_dt_s),
            forward_m=self.forward_window_m,
        )
        self.last_projected_s_m = candidate
        max_advance = max(150.0, 3.0 * float(airspeed_mps) * self.control_dt_s)
        if capture_radius_m > 0.0:
            release_radius = capture_radius_m * capture_release_ratio
            if self.capture_lost:
                self.capture_lost = cte > release_radius or not capture_release_allowed
            else:
                self.capture_lost = cte > capture_radius_m
        else:
            self.capture_lost = False
        if self.capture_lost:
            progress_scale = 0.0
        elif cross_track_scale_m > 0.0:
            progress_scale = 1.0 / (1.0 + (cte / cross_track_scale_m) ** 2)
            progress_scale = float(np.clip(progress_scale, minimum_progress_scale, 1.0))
        else:
            progress_scale = 1.0
        candidate_advance = max(0.0, candidate - self.s_m)
        advance = progress_scale * min(candidate_advance, max_advance)
        self.s_m = float(np.clip(self.s_m + advance, 0.0, self.geometry.total_length_m))
        self.segment_index = max(self.segment_index, index)
        self.last_cte_m = cte
        self.last_progress_scale = progress_scale
        return self.s_m, cte
