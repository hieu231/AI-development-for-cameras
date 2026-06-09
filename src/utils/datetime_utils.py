from datetime import datetime, timezone


def to_local_naive_datetime(value: datetime) -> datetime:
    """Normalize aware datetimes to server-local naive datetimes for DB queries."""
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


def serialize_utc_datetime(value: datetime) -> str:
    """Serialize datetimes as UTC ISO 8601 strings with an explicit +00:00 offset."""
    if value.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        if local_tz is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        value = value.replace(tzinfo=local_tz)

    return value.astimezone(timezone.utc).isoformat()
