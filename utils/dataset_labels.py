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


def abbreviate_dataset_label(label: str) -> str:
    """Shorten dataset labels for display. Shared by train.py and replay_history.py.
    E.g. 'minimal endgames K vs KQ easy' -> 'endgame_KQ_easy', 'minimal_endgames_mate_in_2' -> 'mm2'."""
    if not label:
        return label
    normalized = label.replace("_", " ")
    m = re.search(r"K\s+vs\s+K(Q|R|BB|BN|B|N)\b", normalized, re.I)
    if m:
        suffix = "_easy" if re.search(r"\beasy\b", normalized, re.I) else ""
        return f"endgame_K{m.group(1).upper()}{suffix}"
    for n in (2, 3, 4, 5):
        if re.search(rf"mate[_\s]in[_\s]?{n}\b", normalized, re.I) or f"mate_in_{n}" in label:
            return f"mm{n}"
    return label


def truncate_label(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return f"{label[: max_len - 3]}..."
