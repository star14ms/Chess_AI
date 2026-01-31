"""Helpers to format dataset/theme labels for logging."""
from __future__ import annotations

import re


def strip_dataset_suffixes(base: str) -> str:
    for suffix in ("_flipped", "_aug"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def format_theme_token(token: str) -> str:
    lower = token.lower()
    mate_match = re.search(r"mate[_-]?in[_-]?(\d+)", lower)
    if mate_match:
        return f"m{mate_match.group(1)}"
    return token.replace("_", " ")


def format_theme_group(group: str) -> str:
    parts = [p for p in group.split("-") if p]
    if not parts:
        return group
    formatted = [format_theme_token(part) for part in parts]
    return " + ".join(formatted)


def format_dataset_label(base: str) -> str:
    base = strip_dataset_suffixes(base)
    if "_without_" in base:
        include_raw, exclude_raw = base.split("_without_", 1)
        include_label = format_theme_group(include_raw) if include_raw else "all"
        exclude_label = format_theme_group(exclude_raw) if exclude_raw else ""
        return f"{include_label} (no {exclude_label})" if exclude_label else include_label
    return format_theme_group(base)


def truncate_label(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return f"{label[: max_len - 3]}..."
