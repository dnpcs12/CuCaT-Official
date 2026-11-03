from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .geometry import Vec3, add, dot, lerp, normalize, scale, sub, vec3
from .spec import TargetBodyPreset


@dataclass(frozen=True)
class Bounds3D:
    min: Vec3
    max: Vec3

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "Bounds3D":
        return cls(min=vec3(raw["min"]), max=vec3(raw["max"]))

    def center(self) -> Vec3:
        return (
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        )

    def size(self) -> Vec3:
        return (
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        )

    def corners(self) -> list[Vec3]:
        min_x, min_y, min_z = self.min
        max_x, max_y, max_z = self.max
        return [
            (x, y, z)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
            for z in (min_z, max_z)
        ]

    def translated(self, offset: Vec3) -> "Bounds3D":
        return Bounds3D(add(self.min, offset), add(self.max, offset))

    def shrink_xz(self, factor: float) -> "Bounds3D":
        center = self.center()
        size = self.size()
        half = (size[0] * factor * 0.5, size[1] * 0.5, size[2] * factor * 0.5)
        return Bounds3D(sub(center, half), add(center, half))


@dataclass(frozen=True)
class FrameTargetContext:
    frame: int
    actor_forward: Vec3
    region_bounds: dict[str, Bounds3D]

    def bounds_for(self, preset: TargetBodyPreset) -> Bounds3D:
        direct = self.region_bounds.get(preset.value)
        if direct is not None:
            return direct

        full_body = self.region_bounds.get(TargetBodyPreset.FULL_BODY.value)
        if full_body is None:
            available = ", ".join(sorted(self.region_bounds))
            raise KeyError(f"frame {self.frame} missing full_body bounds. available: {available}")

        min_x, min_y, min_z = full_body.min
        max_x, max_y, max_z = full_body.max
        size_x, size_y, size_z = full_body.size()

        if preset is TargetBodyPreset.FULL_BODY:
            return full_body
        if preset is TargetBodyPreset.LEGS:
            y1 = min_y + size_y * 0.58
            center_x = (min_x + max_x) * 0.5
            center_z = (min_z + max_z) * 0.5
            half_x = size_x * 0.22
            half_z = size_z * 0.20
            return Bounds3D(
                (center_x - half_x, min_y, center_z - half_z),
                (center_x + half_x, y1, center_z + half_z),
            )
        if preset is TargetBodyPreset.FEET:
            y1 = min_y + size_y * 0.20
            center_x = (min_x + max_x) * 0.5
            center_z = (min_z + max_z) * 0.5
            half_x = size_x * 0.24
            half_z = size_z * 0.23
            return Bounds3D(
                (center_x - half_x, min_y, center_z - half_z),
                (center_x + half_x, y1, center_z + half_z),
            )
        if preset is TargetBodyPreset.KNEE_UP:
            y0 = min_y + size_y * 0.27
            return Bounds3D((min_x, y0, min_z), (max_x, max_y, max_z))
        if preset is TargetBodyPreset.ABDOMEN:
            y0 = min_y + size_y * 0.40
            y1 = min_y + size_y * 0.68
            center_x = (min_x + max_x) * 0.5
            center_z = (min_z + max_z) * 0.5
            half_x = size_x * 0.26
            half_z = size_z * 0.20
            return Bounds3D(
                (center_x - half_x, y0, center_z - half_z),
                (center_x + half_x, y1, center_z + half_z),
            )
        if preset is TargetBodyPreset.WAIST_UP:
            y0 = min_y + size_y * 0.45
            return Bounds3D((min_x, y0, min_z), (max_x, max_y, max_z))
        if preset is TargetBodyPreset.BUST_UP:
            y0 = min_y + size_y * 0.72
            center_x = (min_x + max_x) * 0.5
            center_z = (min_z + max_z) * 0.5
            half_x = size_x * 0.56
            half_z = size_z * 0.36
            return Bounds3D(
                (center_x - half_x, y0, center_z - half_z),
                (center_x + half_x, max_y, center_z + half_z),
            )
        if preset is TargetBodyPreset.FACE:
            # Keep the face box tall enough to include more head/hair volume.
            top_padding = size_y * 0.00
            face_height = size_y * 0.24
            y1 = max_y - top_padding
            y0 = max(min_y, y1 - face_height)
            center = ((min_x + max_x) * 0.5, y0 + face_height * 0.5, (min_z + max_z) * 0.5)
            half = (size_x * 0.18, face_height * 0.5, size_z * 0.18)
            return Bounds3D(sub(center, half), add(center, half))

        raise KeyError(f"unsupported region: {preset.value}")


class FrameTargetProvider(Protocol):
    def get_context(self, frame: int) -> FrameTargetContext:
        ...


_FLAT_POSE_FORWARD_RATIO_THRESHOLD = 0.45


def _flat_pose_ratio(context: FrameTargetContext) -> float | None:
    full_body = context.region_bounds.get(TargetBodyPreset.FULL_BODY.value)
    if full_body is None:
        return None
    size_x, size_y, size_z = full_body.size()
    horizontal_span = max(size_x, size_z, 1e-6)
    return size_y / horizontal_span


def _stabilize_actor_forwards(contexts: dict[int, FrameTargetContext]) -> dict[int, FrameTargetContext]:
    """Keep actor forward vectors horizontally normalized and temporally sign-stable.

    Target dumps derive forward from pose bones.  When a character lies down,
    the bone-derived forward can become sign-ambiguous after horizontal
    projection and may flip nearly 180 degrees in one frame.  The shot solver
    interprets that as a real front/back turn.  For very flat poses, the
    head-to-hips axis can also drift into a foot-to-face direction, so keep the
    last reliable upright forward until the body is vertical enough again.
    """

    stable: dict[int, FrameTargetContext] = {}
    previous_forward: Vec3 | None = None
    previous_reliable_forward: Vec3 | None = None
    for frame in sorted(contexts):
        context = contexts[frame]
        flat_ratio = _flat_pose_ratio(context)
        is_flat_pose = (
            flat_ratio is not None
            and flat_ratio < _FLAT_POSE_FORWARD_RATIO_THRESHOLD
            and previous_reliable_forward is not None
        )
        if is_flat_pose:
            forward = previous_reliable_forward
        else:
            fallback = previous_forward or previous_reliable_forward or (0.0, 0.0, 1.0)
            forward = normalize((context.actor_forward[0], 0.0, context.actor_forward[2]), fallback)
            if previous_forward is not None and dot(forward, previous_forward) < 0.0:
                forward = scale(forward, -1.0)
            if flat_ratio is None or flat_ratio >= _FLAT_POSE_FORWARD_RATIO_THRESHOLD:
                previous_reliable_forward = forward
        stable[frame] = FrameTargetContext(
            frame=context.frame,
            actor_forward=forward,
            region_bounds=context.region_bounds,
        )
        previous_forward = forward
    return stable


class JsonFrameTargetProvider:
    """Per-frame body targets read from a target dump.

    ``stabilize_forward`` applies ``_stabilize_actor_forwards``. It is off by default, which is how
    the training labels were reversed; the end-to-end pipeline turns it on.
    """

    def __init__(self, json_path: Path, stabilize_forward: bool = False):
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames_raw = raw.get("frames", [])
        contexts: dict[int, FrameTargetContext] = {}
        for item in frames_raw:
            frame = int(item["frame"])
            bounds = {
                str(name): Bounds3D.from_dict(data)
                for name, data in (item.get("bounds") or {}).items()
            }
            contexts[frame] = FrameTargetContext(
                frame=frame,
                actor_forward=vec3(item.get("forward", [0.0, 0.0, 1.0])),
                region_bounds=bounds,
            )
        self._contexts = _stabilize_actor_forwards(contexts) if stabilize_forward else contexts
        self._frames = sorted(self._contexts)

    def get_context(self, frame: int) -> FrameTargetContext:
        frame = int(frame)
        try:
            return self._contexts[frame]
        except KeyError as exc:
            frames = self._frames
            if not frames:
                raise KeyError("target provider has no frames") from exc
            if frame <= frames[0]:
                return self._contexts[frames[0]]
            if frame >= frames[-1]:
                return self._contexts[frames[-1]]
            pos = bisect_left(frames, frame)
            if pos <= 0 or pos >= len(frames):
                known = ", ".join(str(f) for f in frames)
                raise KeyError(f"frame {frame} not found in target provider. known: {known}") from exc

            left_frame = frames[pos - 1]
            right_frame = frames[pos]
            left_context = self._contexts[left_frame]
            right_context = self._contexts[right_frame]
            span = right_frame - left_frame
            t = 0.0 if span <= 0 else (frame - left_frame) / span

            keys = set(left_context.region_bounds) & set(right_context.region_bounds)
            if not keys:
                raise KeyError(
                    f"frame {frame} cannot be interpolated because surrounding frames "
                    f"{left_frame} and {right_frame} do not share any region bounds"
                ) from exc

            bounds = {}
            for key in keys:
                left_bounds = left_context.region_bounds[key]
                right_bounds = right_context.region_bounds[key]
                bounds[key] = Bounds3D(
                    min=lerp(left_bounds.min, right_bounds.min, t),
                    max=lerp(left_bounds.max, right_bounds.max, t),
                )

            right_forward = right_context.actor_forward
            if dot(left_context.actor_forward, right_forward) < 0.0:
                right_forward = scale(right_forward, -1.0)
            forward = normalize(lerp(left_context.actor_forward, right_forward, t), left_context.actor_forward)
            return FrameTargetContext(
                frame=frame,
                actor_forward=forward,
                region_bounds=bounds,
            )
