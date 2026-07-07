from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
