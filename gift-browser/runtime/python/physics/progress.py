import numpy as np

class Progress:
    """Local projection only: a crossing may not skip future loops/waypoints."""

    def __init__(self, reference, phase):
        self.reference = np.asarray(reference, float)
        self.phase = np.asarray(phase, float)
        self.best = 0.0
        self.last_improvement_s = 0.0
        self.reason = None
        self.endpoint_distance = float('inf')

    def update(self, position, seconds):
        p = np.asarray(position, float)
        if not np.isfinite(p).all():
            self.reason = 'nonfinite_physics'
            return
        start = max(0, np.searchsorted(self.phase, self.best - 200.0) - 1)
        end = min(len(self.phase) - 1, np.searchsorted(self.phase, self.best + 1000.0))
        end = max(start + 1, end)
        (a, b) = (self.reference[start:end], self.reference[start + 1:end + 1])
        delta = b - a
        f = np.clip(np.sum((p - a) * delta, axis=1) / np.maximum(np.sum(delta * delta, axis=1), 1e-20), 0.0, 1.0)
        i = int(np.argmin(np.sum((p - (a + f[:, None] * delta)) ** 2, axis=1)))
        s = self.phase[start + i] + f[i] * (self.phase[start + i + 1] - self.phase[start + i])
        if s > self.best + 2.0:
            self.last_improvement_s = seconds
        self.best = max(self.best, float(s))
        self.endpoint_distance = float(np.linalg.norm(p - self.reference[-1]))
        final_tangent = self.reference[-1] - self.reference[-2]
        crossed = float(np.dot(p - self.reference[-1], final_tangent)) >= 0.0
        if self.best >= self.phase[-1] - 5.0 and (crossed or self.endpoint_distance <= 60.0):
            self.reason = 'route_end'
        elif seconds - self.last_improvement_s >= 180.0:
            self.reason = 'no_geometric_progress'

    @property
    def fraction(self):
        return min(1.0, self.best / self.phase[-1])
