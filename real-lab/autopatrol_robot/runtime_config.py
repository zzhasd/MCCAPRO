"""Portable paths for the optional robot integration."""

import os
from pathlib import Path

import yaml


def config_file_path():
    return Path(os.environ.get(
        "AUTOPATROL_CONFIG", Path(__file__).with_name("config.yaml")
    )).expanduser().resolve()


def load_config(path):
    """Resolve map and image paths relative to the selected configuration file."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for group in ("map_yaml", "image_dir"):
        entries = config.get("paths", {}).get(group, {})
        for key, value in entries.items():
            location = Path(value).expanduser()
            if not location.is_absolute():
                location = path.parent / location
            entries[key] = str(location.resolve())
    return config


def image_directory():
    override = os.environ.get("AUTOPATROL_IMAGE_DIR")
    if override:
        return str(Path(override).expanduser().resolve())
    config = load_config(config_file_path())
    selected = config.get("select_config", 2)
    return config.get("paths", {}).get("image_dir", {}).get(
        selected, str(Path(__file__).resolve().parents[1] / "tempphoto")
    )
