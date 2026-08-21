import numpy as np
import pandas as pd

from leagueplanner import InputParser, LeaguePlanner, PlannerParams


def run_optimizer(
    n_teams: int,
    n_iterations: int = 500,
    input_file: str = "example/input.xlsx",
    sheet_name: str = "LEAGUE A",
    games_per_opponent: int = 2,
) -> np.ndarray | pd.DataFrame:
    input = InputParser(input_file)

    input.from_excel(sheet_name=sheet_name)
    input.data = input.data.iloc[:, : n_teams + 1]  # limit data to n_teams

    input.parse()

    planner = LeaguePlanner(
        input=input,
        params=PlannerParams(
            n_iterations=n_iterations,
            penalties=input.penalties,
            games_per_opponent=games_per_opponent,
        ),
    )
    X, df = planner.optimize()

    if games_per_opponent == 2:
        return X

    return df
