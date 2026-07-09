from __future__ import annotations
import csv, json, os, re
from pathlib import Path
from typing import Any
import yaml

def ensure_dir(path: str | Path) -> Path:
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f) or {}

def write_json(obj: Any, path: str | Path) -> None:
    p=Path(path); ensure_dir(p.parent)
    with open(p, 'w', encoding='utf-8') as f: json.dump(obj, f, indent=2, sort_keys=True)

def append_csv_row(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    p=Path(path); ensure_dir(p.parent); exists=p.exists()
    with open(p, 'a', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames)
        if not exists: w.writeheader()
        w.writerow({k: row.get(k, '') for k in fieldnames})

def resolve_env_path(value: str | None, default: str | None=None) -> str | None:
    if value is None: value=default
    if value is None: return None
    text = str(value)
    def repl(match):
        name, fallback = match.group(1), match.group(2)
        return os.environ.get(name, fallback or '')
    text = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}', repl, text)
    return os.path.expandvars(os.path.expanduser(text))
