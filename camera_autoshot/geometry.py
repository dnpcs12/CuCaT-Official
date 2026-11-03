from __future__ import annotations

import math
from typing import Iterable, TypeAlias


Vec3: TypeAlias = tuple[float, float, float]


def vec3(values: Iterable[float]) -> Vec3:
    x, y, z = values
    return (float(x), float(y), float(z))


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: Vec3, fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    l = length(v)
    if l <= 1e-8:
        return fallback
    inv = 1.0 / l
    return scale(v, inv)


def flatten_xz(v: Vec3, fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    flat = (v[0], 0.0, v[2])
    return normalize(flat, fallback)


def rotate_y(v: Vec3, yaw_deg: float) -> Vec3:
    rad = math.radians(yaw_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


def direction_from_yaw_pitch(base_forward: Vec3, yaw_deg: float, pitch_deg: float) -> Vec3:
    horiz = flatten_xz(base_forward)
    horiz = rotate_y(horiz, yaw_deg)

    pitch_rad = math.radians(pitch_deg)
    c = math.cos(pitch_rad)
    s = math.sin(pitch_rad)
    return normalize((horiz[0] * c, s, horiz[2] * c))


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )
