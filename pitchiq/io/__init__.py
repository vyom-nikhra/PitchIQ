"""Data-source loaders: SoccerNet, Roboflow, StatsBomb open data, Metrica tracking.

Every loader degrades gracefully: if the dataset is not downloaded (or the
API key / NDA password is missing) it raises :class:`DatasetUnavailable` with
download instructions instead of crashing the pipeline.
"""

from pitchiq.io.errors import DatasetUnavailable

__all__ = ["DatasetUnavailable"]
