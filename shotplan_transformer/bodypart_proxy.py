from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PartSpec:
    name: str
    y0: float
    y1: float


@dataclass(frozen=True)
class ProxyCalibration:
    span_scale: float = 1.0
    span_margin: float = 0.0
    down_shift: float = 0.0
    threshold: float = 0.20


PARTS = (
    PartSpec("head", 0.76, 1.00),
    PartSpec("arms_upper_body", 0.58, 0.90),
    # Rough hand/finger proxy. We do not know actual wrist/finger pose from the
    # discrete shot plan, so approximate it as the broad band where hands most
    # often appear in dance poses: lower torso through chest.
    PartSpec("fingers", 0.20, 0.82),
    PartSpec("waist", 0.40, 0.68),
    PartSpec("legs", 0.00, 0.58),
    PartSpec("feet", 0.00, 0.20),
)

BASELINE_PROXY_CALIBRATION = {
    part.name: ProxyCalibration(threshold=0.20)
    for part in PARTS
}

CALIBRATED_PROXY_CALIBRATION = {
    "head": ProxyCalibration(down_shift=0.025, threshold=0.80),
    "arms_upper_body": ProxyCalibration(threshold=0.50),
    "fingers": ProxyCalibration(span_scale=1.15, span_margin=0.025, down_shift=0.02, threshold=0.50),
    "waist": ProxyCalibration(span_scale=1.35, span_margin=0.025, down_shift=0.05, threshold=0.50),
    "legs": ProxyCalibration(span_scale=1.35, span_margin=0.05, down_shift=0.05, threshold=0.50),
    "feet": ProxyCalibration(down_shift=0.075, threshold=0.50),
}

TARGET_ANCHOR_Y_FRACTIONS = {
    "feet": 0.05,
    "lower_leg": 0.22,
    "thigh": 0.38,
    "hip": 0.51,
    "waist": 0.63,
    "chest": 0.76,
    "face": 0.90,
}

FOV_DEGREES = {
    "narrow": 22.0,
    "normal": 30.0,
    "wide": 40.0,
}

EXPORTED_DISTANCE_MULTIPLIERS = {
    "extreme_close": 0.55,
    "very_close": 0.80,
    "close": 1.20,
    "medium_close": 1.75,
    "medium": 2.55,
    "medium_far": 3.70,
    "far": 5.80,
}


def proxy_calibration_for_mode(mode: str) -> dict[str, ProxyCalibration]:
    normalized = str(mode or "calibrated").strip().lower()
    if normalized == "baseline":
        return BASELINE_PROXY_CALIBRATION
    return CALIBRATED_PROXY_CALIBRATION


def proxy_visibility_score(
    *,
    target_anchor: str,
    distance: str,
    fov: str,
    pitch: str,
    part: PartSpec,
    calibration: ProxyCalibration,
) -> float:
    anchor_y = float(TARGET_ANCHOR_Y_FRACTIONS.get(str(target_anchor), TARGET_ANCHOR_Y_FRACTIONS["waist"]))
    distance_multiplier = float(EXPORTED_DISTANCE_MULTIPLIERS.get(str(distance), EXPORTED_DISTANCE_MULTIPLIERS["medium"]))
    fov_deg = float(FOV_DEGREES.get(str(fov), FOV_DEGREES["normal"]))
    half_span = distance_multiplier * math.tan(math.radians(fov_deg) * 0.5)
    if str(pitch) == "top":
        half_span *= 1.25
    elif str(pitch) in {"high", "low"}:
        half_span *= 1.05
    half_span = half_span * float(calibration.span_scale) + float(calibration.span_margin)

    visible_y0 = anchor_y - half_span - float(calibration.down_shift)
    visible_y1 = anchor_y + half_span - float(calibration.down_shift)
    overlap = max(0.0, min(visible_y1, part.y1) - max(visible_y0, part.y0))
    return float(min(1.0, overlap / max(1e-8, part.y1 - part.y0)))


def visible_bodyparts(row: dict, *, proxy_mode: str = "calibrated") -> tuple[str, ...]:
    calibration = proxy_calibration_for_mode(proxy_mode)
    visible: list[str] = []
    for part in PARTS:
        score = proxy_visibility_score(
            target_anchor=str(row.get("target_anchor", "waist")),
            distance=str(row.get("distance", "medium")),
            fov=str(row.get("fov", "normal")),
            pitch=str(row.get("pitch", "level")),
            part=part,
            calibration=calibration[part.name],
        )
        if score >= float(calibration[part.name].threshold):
            visible.append(part.name)
    return tuple(visible)


def bodypart_combo_label(row: dict, *, proxy_mode: str = "calibrated") -> str:
    parts = visible_bodyparts(row, proxy_mode=proxy_mode)
    return "+".join(parts) if parts else "none"
