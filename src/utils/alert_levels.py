from enum import Enum


class AlertLevel(Enum):
    HIGH = "high"
    LOW = "low"


    @classmethod
    def from_value(cls, value, default: "AlertLevel" = None) -> "AlertLevel":
        if default is None:
            default = cls.LOW

        if isinstance(value, cls):
            return value

        if value is None:
            return default

        normalized = str(value).strip().lower()
        if normalized == "critical":
            return cls.HIGH
        for level in cls:
            if level.value == normalized or level.name.lower() == normalized:
                return level
        return default


class AlertColor(Enum):
    HIGH = "#FFA500"      # Orange
    LOW = "#FFFF00"       # Yellow
    GRAY = "#808080"      # Gray (fallback)


# Mapping AlertLevel to AlertColor
ALERT_LEVEL_MAP = {
    AlertLevel.HIGH: AlertColor.HIGH,
    AlertLevel.LOW: AlertColor.LOW,
}
