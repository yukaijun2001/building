"""
# @Time: 2025/6/5 19:08
# @File: __init__.py.py
"""
from pathlib import Path

from .config_loader import ConfigLoader

config = ConfigLoader
config.config_paths = [
    Path(__file__).parent / "default_config.yaml"
]


__all__ = [
    "config"
]