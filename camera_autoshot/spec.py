from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class _TextEnum(str, Enum):
    @classmethod
    def from_text(cls, text: str):
        normalized = (text or "").strip().lower()
        for value, aliases in cls.aliases().items():
            if normalized == value or normalized in aliases:
                return cls(value)
        valid = ", ".join(member.value for member in cls)
        raise ValueError(f"unsupported {cls.__name__}: {text!r}. expected one of: {valid}")

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {}


class TargetBodyPreset(_TextEnum):
    FULL_BODY = "full_body"
    LEGS = "legs"
    FEET = "feet"
    KNEE_UP = "knee_up"
    ABDOMEN = "abdomen"
    WAIST_UP = "waist_up"
    BUST_UP = "bust_up"
    FACE = "face"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.FULL_BODY.value: {"full", "full body"},
            cls.LEGS.value: {"lower body", "leg shot"},
            cls.FEET.value: {"foot", "feet shot", "foot shot", "feet focus", "foot focus"},
            cls.KNEE_UP.value: {"knee up"},
            cls.ABDOMEN.value: {"belly", "stomach", "core", "pelvis focus", "abdomen focus"},
            cls.WAIST_UP.value: {"waist up", "medium body"},
            cls.BUST_UP.value: {"bust up", "upper body", "chest up"},
            cls.FACE.value: {"close face", "head"},
        }


class ShotSizePreset(_TextEnum):
    FAR = "far"
    MEDIUM = "medium"
    CLOSE = "close"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.FAR.value: {"wide", "long shot"},
            cls.MEDIUM.value: {"medium shot"},
            cls.CLOSE.value: {"close up"},
        }


class AnglePreset(_TextEnum):
    FRONT = "front"
    BACK = "back"
    TOP = "top"
    LEFT_45 = "left_45"
    RIGHT_45 = "right_45"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    HIGH_FRONT = "high_front"
    LOW_FRONT = "low_front"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.FRONT.value: {"front center"},
            cls.BACK.value: {"rear", "back center"},
            cls.TOP.value: {"top down", "topdown", "overhead", "birds eye", "bird_eye"},
            cls.LEFT_45.value: {"left 45", "front left"},
            cls.RIGHT_45.value: {"right 45", "front right"},
            cls.PROFILE_LEFT.value: {"profile left", "left side"},
            cls.PROFILE_RIGHT.value: {"profile right", "right side"},
            cls.HIGH_FRONT.value: {"high angle", "high front"},
            cls.LOW_FRONT.value: {"low angle", "low front"},
        }


class CompositionPreset(_TextEnum):
    CENTER = "center"
    LEFT_THIRD = "left_third"
    RIGHT_THIRD = "right_third"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.CENTER.value: {"middle"},
            cls.LEFT_THIRD.value: {"left", "left third"},
            cls.RIGHT_THIRD.value: {"right", "right third"},
        }


class RollPreset(_TextEnum):
    NONE = "none"
    SLIGHT_LEFT = "slight_left"
    LEFT = "left"
    STRONG_LEFT = "strong_left"
    SLIGHT_RIGHT = "slight_right"
    RIGHT = "right"
    STRONG_RIGHT = "strong_right"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.NONE.value: {"no_roll", "level", "upright"},
            cls.SLIGHT_LEFT.value: {"slight left", "small left roll"},
            cls.LEFT.value: {"left roll"},
            cls.STRONG_LEFT.value: {"strong left", "large left roll"},
            cls.SLIGHT_RIGHT.value: {"slight right", "small right roll"},
            cls.RIGHT.value: {"right roll"},
            cls.STRONG_RIGHT.value: {"strong right", "large right roll"},
        }


class TransitionPreset(_TextEnum):
    LINEAR = "linear"
    EASE_IN_OUT = "ease_in_out"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    CUT = "cut"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.LINEAR.value: {"straight"},
            cls.EASE_IN_OUT.value: {"ease in out", "smooth"},
            cls.EASE_IN.value: {"slow start"},
            cls.EASE_OUT.value: {"slow end"},
            cls.CUT.value: {"hard cut"},
        }


@dataclass(frozen=True)
class ShotKeyframeRequest:
    frame: int
    target_body: TargetBodyPreset
    shot_size: ShotSizePreset
    angle: AnglePreset = AnglePreset.FRONT
    composition: CompositionPreset = CompositionPreset.CENTER
    roll: RollPreset = RollPreset.NONE
    transition_to_next: TransitionPreset = TransitionPreset.LINEAR

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ShotKeyframeRequest":
        return cls(
            frame=int(raw["frame"]),
            target_body=TargetBodyPreset.from_text(str(raw.get("target_body", "full_body"))),
            shot_size=ShotSizePreset.from_text(str(raw.get("shot_size", "medium"))),
            angle=AnglePreset.from_text(str(raw.get("angle", "front"))),
            composition=CompositionPreset.from_text(str(raw.get("composition", "center"))),
            roll=RollPreset.from_text(str(raw.get("roll", "none"))),
            transition_to_next=TransitionPreset.from_text(str(raw.get("transition", "linear"))),
        )
