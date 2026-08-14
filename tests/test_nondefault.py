import pandas as pd
import pytest
from utils import optimize


@pytest.mark.parametrize("n_teams", [9])
@pytest.mark.parametrize("games_per_opponent", [1, 3, 5])
def test_odd_games_per_opponent(n_teams, games_per_opponent):
    df = optimize(n_teams, games_per_opponent=games_per_opponent)

    pairs = df.apply(lambda row: tuple(sorted([row["Home"], row["Away"]])), axis=1)
    counts = pairs.value_counts()
    assert (counts == games_per_opponent).all()
    assert len(counts) == n_teams * (n_teams - 1) // 2

    home_counts = df["Home"].value_counts()
    away_counts = df["Away"].value_counts()
    assert ((home_counts + away_counts) == games_per_opponent * (n_teams - 1)).all()
    assert (home_counts - away_counts).abs().max() <= 1


@pytest.mark.parametrize("n_teams", [7])
@pytest.mark.parametrize("games_per_opponent", [4, 6])
def test_even_games_per_opponent(n_teams, games_per_opponent):
    df = optimize(n_teams, games_per_opponent=games_per_opponent)

    pairs = df.apply(lambda row: tuple(sorted([row["Home"], row["Away"]])), axis=1)
    counts = pairs.value_counts()
    assert (counts == games_per_opponent).all()
    assert len(counts) == n_teams * (n_teams - 1) // 2

    home_counts = df["Home"].value_counts()
    away_counts = df["Away"].value_counts()
    expected_home = (games_per_opponent // 2) * (n_teams - 1)
    assert (home_counts == expected_home).all()
    assert (away_counts == expected_home).all()

    scheduled = df.dropna(subset=["Date"]).copy()
    scheduled["Date"] = pd.to_datetime(scheduled["Date"])
    scheduled["pair"] = scheduled.apply(
        lambda row: tuple(sorted([row["Home"], row["Away"]])),
        axis=1,
    )
    pair_gaps = scheduled.sort_values("Date").groupby("pair")["Date"].diff().dt.days
    assert not pair_gaps.dropna().empty
    assert pair_gaps.dropna().min() >= 1
