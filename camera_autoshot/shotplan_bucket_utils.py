from __future__ import annotations

from collections import defaultdict
from typing import Any


def consolidate_frame_bucket(
    rows: list[dict[str, Any]],
    *,
    signature_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Consolidate one sampled-frame bucket.

    Rules:
    1. Remove exact duplicate labels while preserving first occurrence order.
    2. If 3 or more unique labels remain, keep only the chronological first and last.
    """

    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        signature = tuple(row.get(field) for field in signature_fields)
        if signature in seen:
            continue
        seen.add(signature)
        unique_rows.append(row)

    if len(unique_rows) >= 3:
        return [unique_rows[0], unique_rows[-1]]
    return unique_rows


def group_and_consolidate_by_frame(
    rows: list[dict[str, Any]],
    *,
    frame_field: str,
    signature_fields: tuple[str, ...],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[frame_field])].append(row)

    consolidated: dict[int, list[dict[str, Any]]] = {}
    for frame in sorted(grouped.keys()):
        consolidated[int(frame)] = consolidate_frame_bucket(
            grouped[int(frame)],
            signature_fields=signature_fields,
        )
    return consolidated
