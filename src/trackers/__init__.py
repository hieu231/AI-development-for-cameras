# Ultralytics YOLO 🚀, AGPL-3.0 license

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

try:
	from .track import register_tracker
except Exception:  # pragma: no cover - optional for standalone BYTETracker usage
	register_tracker = None

__all__ = 'register_tracker', 'BOTSORT', 'BYTETracker'  # allow simpler import
