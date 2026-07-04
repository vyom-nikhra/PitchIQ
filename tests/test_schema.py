import numpy as np
import pandas as pd
import pytest

from pitchiq.core.schema import (
    BALL_ID,
    MatchMeta,
    homographies_to_frame,
    homography_for_frame,
    load_tracking_table,
    save_tracking_table,
    validate_tracking_table,
)


def _tiny_table():
    return pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "timestamp": [0.0, 0.0, 0.04],
            "entity_id": [1, BALL_ID, 1],
            "class": ["player", "ball", "player"],
            "team": ["home", "none", "home"],
            "jersey_no": [7, None, 7],
            "x_pixel": [10.0, 5.0, 11.0],
            "y_pixel": [20.0, 6.0, 21.0],
            "x_pitch": [50.0, 52.0, 50.5],
            "y_pitch": [30.0, 34.0, 30.2],
            "conf": [0.9, 0.5, 0.9],
        }
    )


def test_roundtrip(tmp_path):
    path = tmp_path / "t.parquet"
    save_tracking_table(_tiny_table(), path)
    df = load_tracking_table(path)
    assert len(df) == 3
    assert df["jersey_no"].dtype == "Int16"
    assert df.loc[df.entity_id == BALL_ID, "jersey_no"].isna().all()


def test_validation_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_tracking_table(pd.DataFrame({"frame": [0]}))


def test_homography_table_roundtrip():
    H = np.arange(9, dtype=float).reshape(3, 3) + 1
    hdf = homographies_to_frame(
        [
            {"frame": 0, "timestamp": 0.0, "H": H, "method": "lines", "reproj_error_px": 1.0},
            {"frame": 1, "timestamp": 0.04, "H": None, "method": "none"},
        ]
    )
    assert np.allclose(homography_for_frame(hdf, 0), H)
    assert homography_for_frame(hdf, 1) is None
    assert homography_for_frame(hdf, 99) is None


def test_meta_attack_sign_halftime(tmp_path):
    meta = MatchMeta(fps=25, n_frames=100, extras={"halftime_frame": 50})
    p = tmp_path / "meta.json"
    meta.save(p)
    m2 = MatchMeta.load(p)
    assert m2.attack_sign("home", 0) == 1
    assert m2.attack_sign("home", 50) == -1
    assert m2.attack_sign("away", 0) == -1
    assert m2.attack_sign("away", 80) == 1
