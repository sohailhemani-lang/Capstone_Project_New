"""Search message content and response fields in the current conversation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldMatch:
    name: str
    value: str


@dataclass(frozen=True)
class HistoryMatch:
    entry_index: int
    role: str
    fields: tuple[FieldMatch, ...]


def _field_values(value: object, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _field_values(nested, child_path)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _field_values(nested, f"{path}[{index}]")
    elif isinstance(value, (str, int, float)):
        yield path, str(value)


def search_history(
    history: Sequence[Mapping[str, object]], query: str
) -> list[HistoryMatch]:
    """Return matching messages, including the names and values of matched fields."""
    needle = query.strip().casefold()
    if not needle:
        return []

    matches = []
    for entry_index, entry in enumerate(history):
        fields = tuple(
            FieldMatch(name, value)
            for name, value in _field_values(entry, "")
            if needle in value.casefold()
        )
        if fields:
            matches.append(
                HistoryMatch(
                    entry_index=entry_index,
                    role=str(entry.get("role", "message")),
                    fields=fields,
                )
            )
    return matches
