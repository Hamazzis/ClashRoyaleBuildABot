"""
Termux assistant mode: overlay + analysis (no auto taps).

Goal:
- Run on Android via Termux + Termux:GUI + Termux:API.
- Show an overlay on top of Clash Royale with basic state and a simple
  placement suggestion ("what to play" and "where to drop").

Notes:
- Screen capture is done via `termux-screenshot` (Termux:API).
- Unit detection (ONNX) is intentionally not used here, because onnxruntime
  is often unavailable on Termux.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable
from typing import Optional

from PIL import Image

from clashroyalebuildabot.constants import SCREENSHOT_HEIGHT
from clashroyalebuildabot.constants import SCREENSHOT_WIDTH
from clashroyalebuildabot.detectors.card_detector import CardDetector
from clashroyalebuildabot.detectors.number_detector import NumberDetector
from clashroyalebuildabot.detectors.screen_detector import ScreenDetector
from clashroyalebuildabot.gui.utils import load_config
from clashroyalebuildabot.namespaces.cards import NAME2CARD
from clashroyalebuildabot.namespaces.cards import Card
from clashroyalebuildabot.namespaces.screens import Screens
from clashroyalebuildabot.utils.logger import setup_logger


@dataclass(frozen=True)
class Suggestion:
    card: Card
    tile_xy: tuple[int, int]
    reason: str


def _resize_for_detectors(img: Image.Image) -> Image.Image:
    # Keep exactly the same input size that the desktop pipeline expects.
    return img.convert("RGB").resize(
        (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.Resampling.BILINEAR
    )


def _termux_screenshot_to(path: str) -> Image.Image:
    """
    Take a screenshot with Termux:API and return a PIL image.

    Requires: `pkg install termux-api` and `termux-screenshot` permission grant.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = subprocess.run(
        ["termux-screenshot", "-f", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg = "termux-screenshot failed."
        if stdout or stderr:
            msg += f" stdout={stdout!r} stderr={stderr!r}"
        raise RuntimeError(msg)
    return Image.open(path)


def _load_deck_from_config(config: dict) -> list[Card]:
    """
    Reads deck from config if present, otherwise falls back to the same 8 cards
    as the desktop `main.py`.

    Config format:
      deck:
        - archers
        - goblin_barrel
        - baby_dragon
        - cannon
        - knight
        - minipekka
        - musketeer
        - witch
    """
    default_names = [
        "archers",
        "goblin_barrel",
        "baby_dragon",
        "cannon",
        "knight",
        "minipekka",
        "musketeer",
        "witch",
    ]
    names: Iterable[str] = config.get("deck") or default_names
    cards: list[Card] = []
    for name in names:
        card = NAME2CARD.get(str(name))
        if card is None:
            raise ValueError(
                f"Unknown card name {name!r} in config deck. "
                f"Expected one of: {', '.join(sorted(NAME2CARD.keys()))}"
            )
        cards.append(card)
    if len(cards) != 8:
        raise ValueError(f"Deck must contain exactly 8 cards, got {len(cards)}.")
    return cards


def _tile_to_screen_xy(tile_x: int, tile_y: int) -> tuple[int, int]:
    """
    Convert tile coordinates (bot's grid) to approximate tap coordinates in the
    720x1280 display space. Useful for human guidance.
    """
    from clashroyalebuildabot.constants import DISPLAY_HEIGHT
    from clashroyalebuildabot.constants import TILE_HEIGHT
    from clashroyalebuildabot.constants import TILE_INIT_X
    from clashroyalebuildabot.constants import TILE_INIT_Y
    from clashroyalebuildabot.constants import TILE_WIDTH

    x = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return int(round(x)), int(round(y))


def _pick_suggestion(cards: list[Card], ready: list[int], elixir: int) -> Optional[Suggestion]:
    """
    Very lightweight heuristic suggestion:
    - Prefer playable, non-spell cards.
    - Prefer cheaper cards (cycle) unless at 10 elixir.
    - Suggest a safe "cycle" drop in the middle of own side.
    """
    # Safe cycle tile (roughly center, own side)
    tile_xy = (8, 9)

    playable: list[Card] = []
    for i in ready:
        # ready indices correspond to crops[1:], so map to cards[1..]
        card = cards[i + 1]
        if card.name == "blank":
            continue
        if card.cost > elixir:
            continue
        playable.append(card)

    if not playable:
        return None

    def score(c: Card) -> float:
        # Penalize spells/anywhere-target cards (no target info in assist mode).
        s = 0.0
        if c.target_anywhere:
            s -= 5.0
        # Prefer cycling cheap cards.
        s -= 0.2 * c.cost
        # If capped on elixir, prefer spending more.
        if elixir >= 9:
            s += 0.15 * c.cost
        return s

    best = max(playable, key=score)
    reason = "cycle/safe drop"
    if best.target_anywhere:
        reason = "spell available (no target detection in assist mode)"
    return Suggestion(card=best, tile_xy=tile_xy, reason=reason)


class _Overlay:
    def __init__(self):
        import termuxgui as tg

        self.tg = tg
        self.conn = tg.Connection()
        self.activity = tg.Activity(self.conn, overlay=True, pip=False)

        self.root = tg.LinearLayout(self.activity, vertical=True)
        self.title = tg.TextView(self.activity, "CRBAB Assist (Termux)", self.root)
        self.title.settextsize(18)
        self.title.setcolor(0, 255, 255, 230)

        self.status = tg.TextView(self.activity, "Starting…", self.root)
        self.status.settextsize(14)
        self.status.setcolor(255, 255, 255, 230)

        self.stop_btn = tg.Button(self.activity, "Stop", self.root)

    def poll_stop(self) -> bool:
        for event in self.activity.events():
            if event.type == self.tg.Event.click:
                if event.value.get("id") == self.stop_btn.id:
                    return True
        return False

    def set_text(self, text: str) -> None:
        self.status.settext(text)

    def close(self) -> None:
        self.activity.finish()
        self.conn.close()


def main() -> None:
    config = load_config()

    # Don't require the desktop GUI logger handler (PyQt).
    config = dict(config or {})
    config.setdefault("bot", {})
    config["bot"]["enable_gui"] = False

    # Keep loguru to stdout/file only.
    setup_logger(main_window=None, config=config)  # type: ignore[arg-type]

    deck = _load_deck_from_config(config)
    card_detector = CardDetector(deck)
    number_detector = NumberDetector()
    screen_detector = ScreenDetector()

    overlay = _Overlay()
    tmp_path = os.path.join(os.getcwd(), ".crbab", "assist_screen.png")
    tick_s = float(config.get("assist", {}).get("tick_seconds", 0.5))

    frame = 0
    last_error: Optional[str] = None

    try:
        while True:
            if overlay.poll_stop():
                break

            try:
                raw = _termux_screenshot_to(tmp_path)
                img = _resize_for_detectors(raw)

                cards, ready = card_detector.run(img)
                numbers = number_detector.run(img)
                screen = screen_detector.run(img)

                elixir = int(numbers.elixir.number)
                hand = cards[1:5]
                hand_str = " | ".join(
                    [
                        f"{'✓' if i in ready else '·'} {c.name}({c.cost})"
                        for i, c in enumerate(hand)
                    ]
                )

                suggestion = None
                if screen == Screens.IN_GAME:
                    suggestion = _pick_suggestion(cards, ready, elixir)

                if screen != Screens.IN_GAME:
                    sug_str = f"Screen: {screen.name} (no in-game advice)"
                elif suggestion is None:
                    sug_str = "No playable cards (or no ready cards)."
                else:
                    px, py = _tile_to_screen_xy(*suggestion.tile_xy)
                    sug_str = (
                        f"Suggest: {suggestion.card.name} @ tile {suggestion.tile_xy} "
                        f"(≈ x={px}, y={py}) — {suggestion.reason}"
                    )

                overlay.set_text(
                    "\n".join(
                        [
                            f"Frame: {frame}",
                            f"Screen: {screen.name}",
                            f"Elixir: {elixir}",
                            f"Hand: {hand_str}",
                            sug_str,
                        ]
                    )
                )
                last_error = None
            except Exception as e:  # keep overlay alive
                last_error = str(e)
                overlay.set_text(
                    "\n".join(
                        [
                            f"Frame: {frame}",
                            "Error:",
                            last_error,
                            "",
                            "Make sure Termux:API is installed and screenshot permission is granted.",
                        ]
                    )
                )

            frame += 1
            time.sleep(tick_s)
    finally:
        overlay.close()


if __name__ == "__main__":
    main()

