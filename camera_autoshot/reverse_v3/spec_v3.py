from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from camera_autoshot.spec import CompositionPreset, RollPreset, TransitionPreset


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


class TargetAnchorPreset(_TextEnum):
    FEET = "feet"
    LOWER_LEG = "lower_leg"
    THIGH = "thigh"
    HIP = "hip"
    WAIST = "waist"
    CHEST = "chest"
    LEGS = "legs"
    TORSO = "torso"
    FACE = "face"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.FEET.value: {"foot", "feet_anchor"},
            cls.LOWER_LEG.value: {"lower leg", "shin", "calf"},
            cls.THIGH.value: {"upper_leg", "upper leg"},
            cls.HIP.value: {"pelvis"},
            cls.WAIST.value: {"belly", "abdomen"},
            cls.CHEST.value: {"bust"},
            cls.LEGS.value: {"leg", "lower_body"},
            cls.TORSO.value: {"body", "upper_body"},
            cls.FACE.value: {"head", "close face"},
        }


class DistancePreset(_TextEnum):
    EXTREME_CLOSE = "extreme_close"
    CLOSE = "close"
    MEDIUM_CLOSE = "medium_close"
    MEDIUM = "medium"
    MEDIUM_FAR = "medium_far"
    FAR = "far"
    WIDE = "wide"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.EXTREME_CLOSE.value: {"xclose", "extreme close"},
            cls.CLOSE.value: {"close up", "closeup"},
            cls.MEDIUM_CLOSE.value: {"medium close", "medium_closeup", "medium close up"},
            cls.MEDIUM.value: {"mid"},
            cls.MEDIUM_FAR.value: {"medium far", "medium wide"},
            cls.FAR.value: {"long"},
            cls.WIDE.value: {"very_far", "wide shot", "full wide"},
        }


class YawPreset(_TextEnum):
    FRONT = "front"
    LEFT_45 = "left_45"
    PROFILE_LEFT = "profile_left"
    BACK_LEFT_45 = "back_left_45"
    BACK = "back"
    BACK_RIGHT_45 = "back_right_45"
    PROFILE_RIGHT = "profile_right"
    RIGHT_45 = "right_45"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.FRONT.value: {"front center"},
            cls.LEFT_45.value: {"front_left", "front left", "left 45"},
            cls.PROFILE_LEFT.value: {"left_profile", "left side", "profile left"},
            cls.BACK_LEFT_45.value: {"rear_left", "back left", "back left 45"},
            cls.BACK.value: {"rear", "back center"},
            cls.BACK_RIGHT_45.value: {"rear_right", "back right", "back right 45"},
            cls.PROFILE_RIGHT.value: {"right_profile", "right side", "profile right"},
            cls.RIGHT_45.value: {"front_right", "front right", "right 45"},
        }


class PitchPreset(_TextEnum):
    LOW = "low"
    LEVEL = "level"
    HIGH = "high"
    TOP = "top"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.LOW.value: {"low_front", "low angle"},
            cls.LEVEL.value: {"front", "normal", "eye_level", "eye level"},
            cls.HIGH.value: {"high_front", "high angle"},
            cls.TOP.value: {"top_down", "top down", "overhead", "bird_eye", "birds eye"},
        }


class FovPreset(_TextEnum):
    NARROW = "narrow"
    NORMAL = "normal"
    WIDE = "wide"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.NARROW.value: {"telephoto", "zoomed"},
            cls.NORMAL.value: {"default"},
            cls.WIDE.value: {"wide_angle", "wide angle"},
        }


class InterpolationPreset(_TextEnum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"

    @classmethod
    def aliases(cls) -> dict[str, set[str]]:
        return {
            cls.LINEAR.value: {"straight"},
            cls.EASE_IN.value: {"slow_start", "slow start"},
            cls.EASE_OUT.value: {"slow_end", "slow end"},
            cls.EASE_IN_OUT.value: {"ease in out", "smooth"},
        }


@dataclass(frozen=True)
class ShotKeyframeRequestV3:
    frame: int
    target_anchor: TargetAnchorPreset
    distance: DistancePreset
    yaw: YawPreset = YawPreset.FRONT
    pitch: PitchPreset = PitchPreset.LEVEL
    composition: CompositionPreset = CompositionPreset.CENTER
    roll: RollPreset = RollPreset.NONE
    fov: FovPreset = FovPreset.NORMAL
    transition_to_next: TransitionPreset = TransitionPreset.LINEAR
    interpolation: InterpolationPreset = InterpolationPreset.LINEAR

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ShotKeyframeRequestV3":
        # Older experimental V3 JSON may contain vertical_composition.  It was
        # removed from the final V3 schema, so keep read compatibility by simply
        # ignoring it here.
        return cls(
            frame=int(raw["frame"]),
            target_anchor=TargetAnchorPreset.from_text(str(raw.get("target_anchor", "chest"))),
            distance=DistancePreset.from_text(str(raw.get("distance", "medium"))),
            yaw=YawPreset.from_text(str(raw.get("yaw", "front"))),
            pitch=PitchPreset.from_text(str(raw.get("pitch", "level"))),
            composition=CompositionPreset.from_text(str(raw.get("composition", "center"))),
            roll=RollPreset.from_text(str(raw.get("roll", "none"))),
            fov=FovPreset.from_text(str(raw.get("fov", "normal"))),
            transition_to_next=TransitionPreset.from_text(str(raw.get("transition", "linear"))),
            interpolation=InterpolationPreset.from_text(str(raw.get("interpolation", "linear"))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": int(self.frame),
            "target_anchor": self.target_anchor.value,
            "distance": self.distance.value,
            "yaw": self.yaw.value,
            "pitch": self.pitch.value,
            "composition": self.composition.value,
            "roll": self.roll.value,
            "fov": self.fov.value,
            "transition": self.transition_to_next.value,
            "interpolation": self.interpolation.value,
        }
