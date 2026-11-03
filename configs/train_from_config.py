"""Run a CuCaT training script with the exact arguments of a released checkpoint.

Usage (from the repository root):
    python configs/train_from_config.py configs/shotplan_v3_1.json
    python configs/train_from_config.py configs/cut_transition.json --epochs 1 --dataset-root path/to/dataset

Every ``--option value`` after the config path overrides the value stored in the config.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_cli(parser: argparse.ArgumentParser, args: dict) -> list[str]:
    actions = {a.dest: a for a in parser._actions if a.option_strings}
    cli: list[str] = []
    for key, value in args.items():
        action = actions.get(key)
        if action is None:
            raise SystemExit(f"Config option {key!r} is not accepted by the training script.")
        flag = next(s for s in action.option_strings if s.startswith("--"))
        kind = type(action).__name__
        if kind == "_StoreTrueAction":
            if value:
                cli.append(flag)
            continue
        if kind == "_StoreFalseAction":
            if not value:
                cli.append(flag)
            continue
        if value is None or value == "":
            continue
        if isinstance(value, list):
            if action.nargs in ("*", "+") or isinstance(action.nargs, int):
                cli += [flag, *(str(v) for v in value)]
                continue
            value = ",".join(str(v) for v in value)
        cli += [flag, str(value)]
    return cli


def _parser_for(module) -> argparse.ArgumentParser:
    if hasattr(module, "build_parser"):
        return module.build_parser()
    # Scripts that only expose parse_args(): capture the parser it builds.
    captured: dict = {}
    original = argparse.ArgumentParser.parse_args

    def _capture(self, *a, **k):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = _capture
    try:
        module.parse_args()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


def build_command(config_path: str | Path, overrides: list[str]) -> list[str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    module = importlib.import_module(config["script"])
    parser = _parser_for(module)
    return [sys.executable, "-m", config["script"], *_to_cli(parser, config["args"]), *overrides]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    command = build_command(sys.argv[1], sys.argv[2:])
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
