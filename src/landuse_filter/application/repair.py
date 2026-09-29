"""One-off repair of published ``generations/`` tables that hold a text more than once.

Earlier publishes re-uploaded a text in every later batch that contained it. The first
occurrence (in file-name order) is kept; files left without rows are dropped.
"""

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def dedupe_generations(files: Sequence[Path]) -> dict[Path, pa.Table | None]:
    """Files whose content changes -> the table to write (``None``: delete the file)."""
    seen: set[str] = set()
    changed: dict[Path, pa.Table | None] = {}
    for path in sorted(files, key=lambda p: p.name):
        table = pq.read_table(path)
        keep = []
        for generation_id in table.column("generation_id").to_pylist():
            keep.append(generation_id not in seen)
            seen.add(generation_id)
        if all(keep):
            continue
        changed[path] = table.filter(pa.array(keep, pa.bool_())) if any(keep) else None
    return changed
