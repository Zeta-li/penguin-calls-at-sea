"""The contest inputs and stable conventions used across the four questions."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "date"
SEED = 20261126
CALLS = ("flat", "mod", "tv")
STIMULI = ("control",) + CALLS
CONTEXTS = ("commuting", "searching", "after")
BEHAVIOURS = ("HeadsUp", "Vocal", "Approach")
FEATURES = ("Duration", "F01", "Fpeak", "Q25", "Q50", "Q75")
METRICS = ("Spectrogram", "Amplitude spectrum", "Envelope")
DISTANCES = np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
STATES = np.array([[int(bit) for bit in f"{i:03b}"] for i in range(8)], dtype=float)
STATE_NAMES = tuple(f"{i:03b}" for i in range(8))
HORIZON = 60.0
REPRESENTATIVE_N = 8
SHARED_RANGE = (20.0, 150.0)


def input_path(index):
    matches = list(SOURCE.glob(f"Data{index:02d}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one Data{index:02d} file, found {matches}")
    return matches[0]


def read_inputs():
    acoustics = pd.read_csv(input_path(1))
    playback = pd.read_csv(input_path(2))
    assert len(acoustics) == 108 and set(acoustics.CallType) == set(CALLS)
    assert not acoustics[list(FEATURES)].isna().any().any()
    assert not acoustics.duplicated().any()
    acoustics["session_date"] = pd.to_datetime(acoustics.Date, format="%d/%m/%Y")
    assert len(playback) == 278 and set(playback.CallType) == set(STIMULI)
    assert set(playback.Context) == set(CONTEXTS)
    assert (playback[["Distance", "Npenguins"]] > 0).all().all()
    counts = pd.crosstab(playback.expID, playback.Type)
    assert len(counts) == 139 and (counts[["call", "control"]] == 1).all().all()
    natural = playback[playback.Type == "call"].set_index("expID")
    playback["pair_track"] = playback.expID.map(natural.PlaybackTrack)
    assert not playback.pair_track.isna().any()
    assert playback.Npenguins.median() == REPRESENTATIVE_N
    group_range = playback.groupby("CallType").Distance.agg(["min", "max"])
    assert tuple(map(float, (group_range["min"].max(), group_range["max"].min()))) == SHARED_RANGE
    flags = event_flags(playback)
    assert np.array_equal(playback.ResponseYesNo.to_numpy(), flags.any(axis=1))
    onset = latency_values(playback)
    assert np.isfinite(onset[flags]).all()
    assert ((0 <= onset[flags]) & (onset[flags] <= HORIZON)).all()

    correlation = np.empty((len(CALLS), len(METRICS), 300, len(DISTANCES)))
    for index in range(3, 12):
        metric, call = divmod(index - 3, len(CALLS))
        table = pd.read_csv(input_path(index))
        assert tuple(table.columns) == tuple(f"{d:g}m" for d in DISTANCES)
        values = table.to_numpy(float)
        assert values.shape == (300, 7) and np.nanmax(np.abs(values)) <= 1
        correlation[call, metric] = values
    return acoustics, playback, correlation


def event_flags(playback, time_priority=False):
    values = playback[[f"{b}YesNo" for b in BEHAVIOURS]].to_numpy()
    assert np.isin(values, [0, 1]).all()
    flags = values.astype(bool)
    return flags | (latency_values(playback) > 0) if time_priority else flags


def latency_values(playback):
    return playback[[f"{b}Latency" for b in BEHAVIOURS]].to_numpy(float)


def state_index(flags):
    return np.asarray(flags, dtype=int) @ np.array([4, 2, 1])


def context_weights(playback):
    context = playback.drop_duplicates("expID").Context.value_counts()
    return context.reindex(CONTEXTS).to_numpy() / context.sum()
