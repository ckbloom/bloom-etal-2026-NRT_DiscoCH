"""
config.py

Machine-specific paths (FORCE data, validation polygons, boundaries, the
2018 mosaics, ...) live in config.yaml at the project root, which is not
committed. config.example.yaml is the committed template: copy it to
config.yaml and fill in your own paths.

  - Relative paths are resolved against the project root, so the defaults
    (data/..., output) work wherever the repository is checked out.
  - ~ and environment variables (%VAR% / $VAR) are expanded.
  - An empty value (or ~ / null) becomes None, for data you don't have;
    asking for it with require() gives a clear error.
  - The environment variable DISCOCH_CONFIG points to another config file
    (e.g. a second machine or an operational server).

Usage (notebook or script):

    from src.disco_ch_operational.config import load_config

    cfg = load_config()
    force_directory = cfg["force_dir"]
    districts = cfg.require("districts")   # error if it isn't set
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.yaml"
EXAMPLE_FILE = ROOT / "config.example.yaml"
ENV_VAR = "DISCOCH_CONFIG"


class Config(dict):
    """dict of resolved paths (str or None) with require() for mandatory keys."""

    def __init__(self, values, source):
        super().__init__(values)
        self.source = source

    def __missing__(self, key):
        raise KeyError(f"'{key}' is not in {self.source} -- add it (see {EXAMPLE_FILE.name})")

    def require(self, key):
        value = self[key]
        if value is None:
            raise ValueError(f"'{key}' is empty in {self.source} -- set it to a path on this machine")
        return value


def _resolve(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        return value  # numbers / booleans / lists pass through
    path = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not path.is_absolute() and not value.strip().startswith(("\\\\", "//")):
        path = ROOT / path
    return str(path)


def load_config(path=None):
    """Read config.yaml (or `path`, or $DISCOCH_CONFIG) into a Config of
    resolved paths."""
    import yaml

    path = Path(path or os.environ.get(ENV_VAR) or CONFIG_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"No config file at {path}.\n"
            f"Copy {EXAMPLE_FILE.name} to {CONFIG_FILE.name} in {ROOT} and set the paths for this machine.")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config({k: _resolve(v) for k, v in raw.items()}, source=path)
