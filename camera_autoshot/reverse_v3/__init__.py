"""Reverse V3 experimental shot-plan format.

This package is intentionally separate from the baseline reverse/solver
modules so existing pipelines keep running while V3 is tuned.
"""

from .spec_v3 import (
    DistancePreset,
    FovPreset,
    InterpolationPreset,
    PitchPreset,
    ShotKeyframeRequestV3,
    TargetAnchorPreset,
    YawPreset,
)

__all__ = [
    "DistancePreset",
    "FovPreset",
    "InterpolationPreset",
    "PitchPreset",
    "ShotKeyframeRequestV3",
    "TargetAnchorPreset",
    "YawPreset",
]
