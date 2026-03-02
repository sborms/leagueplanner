import numpy as np

from leagueplanner import InputParser, LeaguePlanner, PlannerParams


def optimize(
    n_teams: int,
    n_iterations: int,
    input_file: str = "example/input.xlsx",
    sheet_name: str = "LEAGUE A",
) -> np.ndarray:
    input = InputParser(input_file)

    input.from_excel(sheet_name=sheet_name)
    input.parse()

    input.data = input.data.iloc[:, : n_teams + 1]  # limit data to n_teams

    planner = LeaguePlanner(
        input=input,
        params=PlannerParams(n_iterations=n_iterations, penalties=input.penalties),
    )
    planner.construction_phase()
    planner.tabu_phase()

    return planner.X
