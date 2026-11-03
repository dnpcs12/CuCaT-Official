from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .geometry import Vec3
from .solver import SolvedCameraKey
from .spec import TransitionPreset


HEADER_TEXT = b"Vocaloid Motion Data 0002"
HEADER_SIZE = 30
MODEL_NAME_SIZE = 20
CAMERA_RECORD_SIZE = 61


TRANSITION_BYTES = {
    TransitionPreset.LINEAR: (20, 107, 20, 107),
    TransitionPreset.EASE_IN_OUT: (32, 96, 32, 96),
    TransitionPreset.EASE_IN: (64, 107, 0, 96),
    TransitionPreset.EASE_OUT: (20, 72, 32, 127),
}


INTERPOLATION_BYTES_BY_NAME = {
    "linear": TRANSITION_BYTES[TransitionPreset.LINEAR],
    "ease_in_out": TRANSITION_BYTES[TransitionPreset.EASE_IN_OUT],
    "ease_in": TRANSITION_BYTES[TransitionPreset.EASE_IN],
    "ease_out": TRANSITION_BYTES[TransitionPreset.EASE_OUT],
}


@dataclass(frozen=True)
class CameraVmdKey:
    frame: int
    distance: float
    interest: Vec3
    rotate: Vec3
    view_angle_deg: int
    perspective: bool
    interpolation: bytes


@dataclass(frozen=True)
class EvaluatedCamera:
    frame: float
    interest: Vec3
    rotate: Vec3
    distance: float
    view_angle_deg: float
    perspective: bool


@dataclass(frozen=True)
class VmdBezier:
    cp1: tuple[float, float]
    cp2: tuple[float, float]

    def eval_x(self, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        it = 1.0 - t
        it2 = it * it
        it3 = it2 * it
        return t3 + 3.0 * t2 * it * self.cp2[0] + 3.0 * t * it2 * self.cp1[0]

    def eval_y(self, t: float) -> float:
        t2 = t * t
        t3 = t2 * t
        it = 1.0 - t
        it2 = it * it
        it3 = it2 * it
        return t3 + 3.0 * t2 * it * self.cp2[1] + 3.0 * t * it2 * self.cp1[1]

    def find_bezier_x(self, time: float) -> float:
        epsilon = 0.00001
        start = 0.0
        stop = 1.0
        t = 0.5
        x = self.eval_x(t)
        while abs(time - x) > epsilon:
            if time < x:
                stop = t
            else:
                start = t
            t = (stop + start) * 0.5
            x = self.eval_x(t)
        return t


def _fixed_string(text: str, size: int) -> bytes:
    raw = text.encode("shift_jis", errors="replace")[:size]
    return raw + (b"\x00" * (size - len(raw)))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp3(a: Vec3, b: Vec3, tx: float, ty: float, tz: float) -> Vec3:
    return (
        _lerp(a[0], b[0], tx),
        _lerp(a[1], b[1], ty),
        _lerp(a[2], b[2], tz),
    )


def _camera_beziers(interpolation: bytes) -> tuple[VmdBezier, VmdBezier, VmdBezier, VmdBezier, VmdBezier, VmdBezier]:
    if len(interpolation) < 24:
        interpolation = bytes(interpolation) + bytes(max(0, 24 - len(interpolation)))

    def make(offset: int) -> VmdBezier:
        x0 = interpolation[offset + 0] / 127.0
        x1 = interpolation[offset + 1] / 127.0
        y0 = interpolation[offset + 2] / 127.0
        y1 = interpolation[offset + 3] / 127.0
        return VmdBezier((x0, y0), (x1, y1))

    return (
        make(0),
        make(4),
        make(8),
        make(12),
        make(16),
        make(20),
    )


def _transition_cp(transition: TransitionPreset) -> tuple[int, int, int, int]:
    if transition is TransitionPreset.CUT:
        transition = TransitionPreset.LINEAR
    return TRANSITION_BYTES[transition]


def _curve_cp(curve: object, transition: TransitionPreset) -> tuple[int, int, int, int]:
    if transition is TransitionPreset.CUT:
        return TRANSITION_BYTES[TransitionPreset.LINEAR]
    value = getattr(curve, "value", curve)
    if value is None:
        return _transition_cp(transition)
    return INTERPOLATION_BYTES_BY_NAME.get(str(value), _transition_cp(transition))


def _camera_interpolation_bytes(
    transition: TransitionPreset,
    curve: object | None = None,
    linear_translation: bool = False,
) -> bytes:
    other_cp = _curve_cp(curve, transition)
    translation_cp = _transition_cp(TransitionPreset.LINEAR) if linear_translation else other_cp
    channels = [
        translation_cp,
        translation_cp,
        translation_cp,
        other_cp,
        other_cp,
        other_cp,
    ]
    payload = bytearray()
    for cp in channels:
        payload.extend(cp)
    return bytes(payload)


def _materialize_keys(
    keys: list[SolvedCameraKey],
    linear_translation: bool = False,
) -> list[CameraVmdKey]:
    if not keys:
        return []

    expanded: list[SolvedCameraKey] = []
    for index, key in enumerate(keys):
        expanded.append(key)
        if index + 1 >= len(keys):
            continue

        next_key = keys[index + 1]
        if key.transition_to_next is TransitionPreset.CUT and next_key.frame - key.frame > 1:
            expanded.append(
                replace(
                    key,
                    frame=next_key.frame - 1,
                    transition_to_next=TransitionPreset.LINEAR,
                )
            )

    expanded.sort(key=lambda item: item.frame)

    materialized: list[CameraVmdKey] = []
    for index, key in enumerate(expanded):
        transition = key.transition_to_next if index + 1 < len(expanded) else TransitionPreset.LINEAR
        curve = getattr(key, "interpolation_to_next", None) if index + 1 < len(expanded) else None
        materialized.append(
            CameraVmdKey(
                frame=key.frame,
                distance=key.distance,
                interest=key.interest,
                rotate=key.rotate,
                view_angle_deg=key.view_angle_deg,
                perspective=key.perspective,
                interpolation=_camera_interpolation_bytes(
                    transition,
                    curve=curve,
                    linear_translation=linear_translation,
                ),
            )
        )
    return materialized


def write_camera_vmd(
    keys: list[SolvedCameraKey],
    output_path: Path,
    model_name: str = "Autoshot Camera",
    linear_translation: bool = False,
) -> Path:
    camera_keys = _materialize_keys(keys, linear_translation=linear_translation)
    return write_camera_vmd_records(camera_keys, output_path, model_name=model_name)


def write_camera_vmd_records(
    camera_keys: list[CameraVmdKey],
    output_path: Path,
    model_name: str = "Autoshot Camera",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = bytearray()
    payload.extend(_fixed_string(HEADER_TEXT.decode("ascii"), HEADER_SIZE))
    payload.extend(_fixed_string(model_name, MODEL_NAME_SIZE))

    payload.extend(struct.pack("<I", 0))
    payload.extend(struct.pack("<I", 0))
    payload.extend(struct.pack("<I", len(camera_keys)))

    for key in camera_keys:
        payload.extend(struct.pack("<I", key.frame))
        payload.extend(struct.pack("<f", key.distance))
        payload.extend(struct.pack("<3f", key.interest[0], key.interest[1], -key.interest[2]))
        payload.extend(struct.pack("<3f", key.rotate[0], key.rotate[1], key.rotate[2]))
        payload.extend(key.interpolation)
        payload.extend(struct.pack("<I", int(key.view_angle_deg)))
        payload.extend(struct.pack("<B", 0 if key.perspective else 1))

    payload.extend(struct.pack("<I", 0))
    payload.extend(struct.pack("<I", 0))
    payload.extend(struct.pack("<I", 0))

    output_path.write_bytes(payload)
    return output_path


def read_camera_vmd_stats(vmd_path: Path) -> dict[str, int]:
    data = Path(vmd_path).read_bytes()
    offset = HEADER_SIZE + MODEL_NAME_SIZE

    motion_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + motion_count * 111

    morph_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + morph_count * 23

    camera_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    max_frame = 0
    for i in range(camera_count):
        record_offset = offset + i * CAMERA_RECORD_SIZE
        frame = struct.unpack_from("<I", data, record_offset)[0]
        max_frame = max(max_frame, frame)

    return {"camera_count": camera_count, "camera_max_frame": max_frame}


def read_camera_vmd(vmd_path: Path) -> list[CameraVmdKey]:
    data = Path(vmd_path).read_bytes()
    offset = HEADER_SIZE + MODEL_NAME_SIZE

    motion_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + motion_count * 111

    morph_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4 + morph_count * 23

    camera_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    keys: list[CameraVmdKey] = []
    for i in range(camera_count):
        record_offset = offset + i * CAMERA_RECORD_SIZE
        frame = struct.unpack_from("<I", data, record_offset)[0]
        distance = struct.unpack_from("<f", data, record_offset + 4)[0]
        interest_x, interest_y, interest_z = struct.unpack_from("<3f", data, record_offset + 8)
        rotate = struct.unpack_from("<3f", data, record_offset + 20)
        interpolation = data[record_offset + 32 : record_offset + 56]
        view_angle_deg = struct.unpack_from("<I", data, record_offset + 56)[0]
        perspective_flag = struct.unpack_from("<B", data, record_offset + 60)[0]
        keys.append(
            CameraVmdKey(
                frame=frame,
                distance=distance,
                interest=(interest_x, interest_y, -interest_z),
                rotate=(rotate[0], rotate[1], rotate[2]),
                view_angle_deg=int(view_angle_deg),
                perspective=(perspective_flag == 0),
                interpolation=bytes(interpolation),
            )
        )
    keys.sort(key=lambda item: item.frame)
    return keys


def evaluate_camera_vmd(keys: list[CameraVmdKey], frame: float) -> EvaluatedCamera | None:
    if not keys:
        return None

    bound_index = None
    frame_i = int(frame)
    for index, key in enumerate(keys):
        if key.frame > frame_i:
            bound_index = index
            break

    if bound_index is None:
        select_key = keys[-1]
        return EvaluatedCamera(
            frame=frame,
            interest=select_key.interest,
            rotate=select_key.rotate,
            distance=select_key.distance,
            view_angle_deg=float(select_key.view_angle_deg),
            perspective=select_key.perspective,
        )

    select_key = keys[bound_index]
    if bound_index == 0:
        return EvaluatedCamera(
            frame=frame,
            interest=select_key.interest,
            rotate=select_key.rotate,
            distance=select_key.distance,
            view_angle_deg=float(select_key.view_angle_deg),
            perspective=select_key.perspective,
        )

    key0 = keys[bound_index - 1]
    key1 = keys[bound_index]
    if (key1.frame - key0.frame) <= 1:
        return EvaluatedCamera(
            frame=frame,
            interest=key0.interest,
            rotate=key0.rotate,
            distance=key0.distance,
            view_angle_deg=float(key0.view_angle_deg),
            perspective=key0.perspective,
        )

    time_range = float(key1.frame - key0.frame)
    time = (frame - float(key0.frame)) / time_range
    ix_bezier, iy_bezier, iz_bezier, rotate_bezier, distance_bezier, fov_bezier = _camera_beziers(key0.interpolation)

    ix_x = ix_bezier.find_bezier_x(time)
    iy_x = iy_bezier.find_bezier_x(time)
    iz_x = iz_bezier.find_bezier_x(time)
    rotate_x = rotate_bezier.find_bezier_x(time)
    distance_x = distance_bezier.find_bezier_x(time)
    fov_x = fov_bezier.find_bezier_x(time)

    ix_y = ix_bezier.eval_y(ix_x)
    iy_y = iy_bezier.eval_y(iy_x)
    iz_y = iz_bezier.eval_y(iz_x)
    rotate_y = rotate_bezier.eval_y(rotate_x)
    distance_y = distance_bezier.eval_y(distance_x)
    fov_y = fov_bezier.eval_y(fov_x)

    interest = _lerp3(key0.interest, key1.interest, ix_y, iy_y, iz_y)
    rotate = (
        _lerp(key0.rotate[0], key1.rotate[0], rotate_y),
        _lerp(key0.rotate[1], key1.rotate[1], rotate_y),
        _lerp(key0.rotate[2], key1.rotate[2], rotate_y),
    )
    distance = _lerp(key0.distance, key1.distance, distance_y)
    view_angle_deg = _lerp(float(key0.view_angle_deg), float(key1.view_angle_deg), fov_y)

    return EvaluatedCamera(
        frame=frame,
        interest=interest,
        rotate=rotate,
        distance=distance,
        view_angle_deg=view_angle_deg,
        perspective=key1.perspective,
    )


def classify_camera_transition(interpolation: bytes) -> TransitionPreset:
    if len(interpolation) < 24:
        return TransitionPreset.LINEAR

    # Translation occupies the first 3 channels, then rotation / distance / fov.
    # For autoshot-generated VMDs, the latter 3 channels carry the intended preset.
    channel = tuple(int(b) for b in interpolation[12:16])
    for transition, cp in TRANSITION_BYTES.items():
        if channel == cp:
            return transition

    translation_channel = tuple(int(b) for b in interpolation[0:4])
    for transition, cp in TRANSITION_BYTES.items():
        if translation_channel == cp:
            return transition
    return TransitionPreset.LINEAR


def dump_debug_json(keys: list[SolvedCameraKey], output_path: Path) -> Path:
    rows = []
    for key in keys:
        row = asdict(key)
        row["transition_to_next"] = key.transition_to_next.value
        rows.append(row)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
