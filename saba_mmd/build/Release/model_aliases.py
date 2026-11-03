from __future__ import annotations

from pathlib import Path

# Optional short names for --model-alias, e.g. {"your_model": Path(r"D:\models\your_model\your_model.pmx")}.
MODEL_ALIASES: dict[str, Path] = {}


def resolve_model_choice(model_path: str = "", model_alias: str = "", default: Path | None = None) -> Path:
    if model_path:
        return Path(model_path)
    if model_alias:
        try:
            return MODEL_ALIASES[model_alias]
        except KeyError as e:
            known = ", ".join(sorted(MODEL_ALIASES)) or "none; add them to MODEL_ALIASES in model_aliases.py"
            raise KeyError(f"unknown model alias: {model_alias} (known: {known})") from e
    if default is None:
        raise ValueError("default model path is required when model_path and model_alias are empty")
    return default
