"""
Top-level exports for `clashroyalebuildabot`.

This project targets desktop (PyQt6 + ADB + ONNX) but we also want to be able to
import lightweight parts (e.g., detectors that don't require ADB / PyQt / ONNX)
in constrained environments like Termux.

To avoid import-time crashes when optional dependencies are missing, heavy
objects are exposed via lazy imports (PEP 562).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import constants as constants
from .namespaces import Cards
from .namespaces import Screens
from .namespaces import State
from .namespaces import Units

__all__ = [
    "constants",
    "Visualizer",
    "Cards",
    "Units",
    "State",
    "Detector",
    "OnnxDetector",
    "ScreenDetector",
    "NumberDetector",
    "UnitDetector",
    "CardDetector",
    "Emulator",
    "Bot",
]

_LAZY_IMPORTS: dict[str, str] = {
    # GUI / desktop-only components
    "Visualizer": "clashroyalebuildabot.visualizer:Visualizer",
    "Emulator": "clashroyalebuildabot.emulator.emulator:Emulator",
    "Bot": "clashroyalebuildabot.bot.bot:Bot",
    # Detectors (some may have optional deps)
    "Detector": "clashroyalebuildabot.detectors.detector:Detector",
    "OnnxDetector": "clashroyalebuildabot.detectors.onnx_detector:OnnxDetector",
    "ScreenDetector": "clashroyalebuildabot.detectors.screen_detector:ScreenDetector",
    "NumberDetector": "clashroyalebuildabot.detectors.number_detector:NumberDetector",
    "UnitDetector": "clashroyalebuildabot.detectors.unit_detector:UnitDetector",
    "CardDetector": "clashroyalebuildabot.detectors.card_detector:CardDetector",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr = target.split(":")
    module = import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value  # cache for future access
    return value

