"""Shared loader errors."""


class DatasetUnavailable(RuntimeError):
    """Raised when an external dataset is not present locally.

    The message always includes how to obtain the data, so callers can surface
    it directly to users.
    """
