import logging

import numpy as np
import pandas as pd


def ndigits(number: float) -> int:
    """Returns number of digits in a number."""
    return len(str(number))


def fill_value(x, unavailable: str) -> int:
    """Returns -1 if 'unavailable', 0 if NaN, and 1 if any other string."""
    if x == unavailable:
        return -1
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        return 1


def setup_logger(logfile: str) -> logging.Logger:
    """Sets up logger with file and console/terminal handlers."""
    logger = logging.getLogger(__name__)

    # remove existing handlers first
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def gather_stats(d_val: dict, d_stats: dict = None) -> dict:
    """Updates statistics dictionary from validation dictionary."""
    if d_stats is None:
        keys = [
            "cost",
            "teams",
            "games",
            "unscheduled",
            "missing_home_slots",
            "min_gap_pairs",
            "max_rest_days",
            "n_high_rest_days_all",
            "n_high_rest_days_rel",
        ]
        d_stats = {key: [] for key in keys}

    for key in d_stats.keys():
        d_stats[key].append(d_val[key])

    return d_stats


def drop_nearby_points_from_array(arr: np.array, r_max: int) -> np.array:
    """Sequentially drops second point if pairwise difference is less than r_max."""
    if len(arr) <= 1:
        return arr.copy()

    arr_out = [arr[0]]
    for i in range(1, len(arr)):
        val = arr[i]
        if (val - arr_out[-1]) + 1 < r_max:
            continue
        else:
            arr_out.append(val)

    return np.array(arr_out)


def get_feasible_home_slots(sets_home: dict[int, set[int]], r_max: int) -> dict:
    """Returns a dictionary with feasible home slots per team."""
    # NOTE: This deals with teams providing more than one home slot within
    # 'r_max' slots by simply dropping the first slot per team
    feasible_home_slots = {}
    for team_idx, set_home in sets_home.items():
        feasible_home_slots[team_idx] = drop_nearby_points_from_array(set_home, r_max)
    return feasible_home_slots


def get_homeless_teams(d_slots: dict) -> list:
    """Returns those teams that have no home slots."""
    teams_without_home_slots = []
    for team_idx, set_home in d_slots.items():
        if len(set_home) == 0:
            teams_without_home_slots.append(team_idx)
    return teams_without_home_slots
