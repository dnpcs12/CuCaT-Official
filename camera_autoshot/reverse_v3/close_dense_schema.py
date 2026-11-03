from __future__ import annotations

from camera_autoshot.reverse_v3.solver_v3 import DISTANCE_HEIGHT_MULTIPLIERS
from camera_autoshot.reverse_v3.spec_v3 import DistancePreset


EXPORTED_DISTANCE_VALUES = (
    "extreme_close",
    "very_close",
    "close",
    "medium_close",
    "medium",
    "medium_far",
    "far",
)


# The original V3 enum is preserved for compatibility. V3.1 reuses the internal
# `wide` enum slot as the closest exported bucket and remaps exported labels only
# when solving to VMD.
EXPORT_TO_INTERNAL_DISTANCE = {
    "extreme_close": "wide",
    "very_close": "extreme_close",
    "close": "close",
    "medium_close": "medium_close",
    "medium": "medium",
    "medium_far": "medium_far",
    "far": "far",
}


INTERNAL_TO_EXPORT_DISTANCE = {
    "wide": "extreme_close",
    "extreme_close": "very_close",
    "close": "close",
    "medium_close": "medium_close",
    "medium": "medium",
    "medium_far": "medium_far",
    "far": "far",
}


CLOSE_DENSE_INTERNAL_DISTANCE_MULTIPLIERS = {
    DistancePreset.WIDE: 0.55,
    DistancePreset.EXTREME_CLOSE: 0.80,
    DistancePreset.CLOSE: 1.20,
    DistancePreset.MEDIUM_CLOSE: 1.75,
    DistancePreset.MEDIUM: 2.55,
    DistancePreset.MEDIUM_FAR: 3.70,
    DistancePreset.FAR: 5.80,
}


def uses_close_dense_distance_schema(payload: dict[str, object]) -> bool:
    """Return True when a V3/V3.1 plan payload uses exported close-dense labels."""

    schema = str(payload.get("schema", "")).lower()
    if "close_dense" in schema or payload.get("distance_schema") is not None:
        return True
    rows = payload.get("keyframes", [])
    if isinstance(rows, list):
        return any(isinstance(row, dict) and str(row.get("distance", "")) == "very_close" for row in rows)
    return False


def apply_close_dense_distance_schema() -> None:
    """Patch process-local V3 distance multipliers for close-dense V3.1 solving."""

    for preset, value in CLOSE_DENSE_INTERNAL_DISTANCE_MULTIPLIERS.items():
        DISTANCE_HEIGHT_MULTIPLIERS[preset] = float(value)


def apply_close_dense_distance_schema_from_payload(payload: dict[str, object]) -> None:
    """Patch V3 distance multipliers, preferring values embedded in a plan JSON.

    V3.1.1 keeps the same exported labels as V3.1 but changes the physical
    multipliers for a few close-range buckets. Storing those values in the JSON
    makes the version self-describing instead of relying on a process-local
    monkey patch.
    """

    apply_close_dense_distance_schema()
    raw_schema = payload.get("distance_schema")
    if not isinstance(raw_schema, dict):
        return
    raw_multipliers = raw_schema.get("body_height_multipliers")
    if not isinstance(raw_multipliers, dict):
        return
    for exported_label, raw_value in raw_multipliers.items():
        internal_label = EXPORT_TO_INTERNAL_DISTANCE.get(str(exported_label), str(exported_label))
        try:
            preset = DistancePreset(internal_label)
            DISTANCE_HEIGHT_MULTIPLIERS[preset] = float(raw_value)
        except (TypeError, ValueError):
            continue


def internalize_close_dense_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one exported V3.1 row into the internal legacy enum labels."""

    copied = dict(row)
    distance = str(copied.get("distance", ""))
    copied["distance"] = EXPORT_TO_INTERNAL_DISTANCE.get(distance, distance)
    return copied
