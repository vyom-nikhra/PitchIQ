"""Layer 2 — Analytics: pure dataframe analysis over the tracking table.

Every module here consumes the tracking table (+ MatchMeta) and returns plain
dicts / DataFrames / arrays; nothing touches video. The
:class:`pitchiq.pipeline.analytics.AnalyticsPipeline` orchestrates and
persists everything through the ArtifactStore.
"""
