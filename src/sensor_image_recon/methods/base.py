from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class MethodSpec:
    name: str
    train: Callable[[dict], Path]
    infer: Callable[[Path | str, Path | str | None], Path]
    evaluate: Callable[[Path | str], dict]
