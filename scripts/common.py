"""Shared helper: resolve the dataset config to an absolute path (Ultralytics would otherwise
resolve the relative `path:` against its own global datasets directory)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def data_yaml() -> str:
    cfg = yaml.safe_load((ROOT / "configs" / "african-wildlife.yaml").read_text())
    cfg["path"] = str((ROOT / "configs" / cfg["path"]).resolve())
    out = ROOT / "runs" / "data_resolved.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)
