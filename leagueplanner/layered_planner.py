import copy
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .constants import LARGE_NBR, OUTPUT_COLS
from .input_parser import InputParser
from .params import PlannerParams
from .solver import Solver


@dataclass
class LayeredPlannerResult:
    calendar: pd.DataFrame
    list_full_costs: list[float]
    used_home_slots: dict[int, set[int]]


class LayeredPlanner:
    """Manages the Solver in a layered fashion for the non-default case of games_per_opponent != 2."""

    def __init__(
        self,
        input: InputParser,
        params: PlannerParams,
        logger: logging.Logger = logging.getLogger(__name__),
        *,
        has_odd_layer: bool,
        n_layers_full: int,
        home_maps: list[dict[int, set[int]]],
    ) -> None:
        """See LeaguePlanner for details."""
        self.input = input
        self.params = params
        self.logger = logger
        self.has_odd_layer = has_odd_layer
        self.home_maps = home_maps

        # precomputed variables
        self.teams = self.input.sets["teams"]
        self.n_layers = n_layers_full + int(self.has_odd_layer)
        self._team_idx_by_name = {
            team_name: team_idx
            for team_idx, team_name in self.input.sets["teams"].items()
        }
        self._slot_by_date = {
            pd.Timestamp(date): slot for slot, date in self.input.sets["slots"].items()
        }

    def run(
        self, progress_bar: st.delta_generator.DeltaGenerator = None
    ) -> LayeredPlannerResult:
        col_date = OUTPUT_COLS[0]
        col_home = OUTPUT_COLS[3]
        col_away = OUTPUT_COLS[4]

        played_slots_by_team = {team_idx: set() for team_idx in self.teams}
        used_home_slots = {team_idx: set() for team_idx in self.teams}

        unused_home_slots = {
            team_idx: {
                self.input.sets["slots"][s]
                for s in set(self.input.sets["home"][team_idx]).difference(
                    used_home_slots[team_idx]
                )
            }
            for team_idx in self.teams
        }

        list_full_costs, frames = [], []
        for layer_idx, home_map in enumerate(self.home_maps):
            self.logger.info(f"** Running layer {layer_idx + 1}/{self.n_layers} **")

            layer_input = self._prepare_layer_input(
                played_slots_by_team=played_slots_by_team,
            )

            layer_params = copy.deepcopy(self.params)
            layer_params.games_per_opponent = 2

            solver = Solver(
                input=layer_input,
                params=layer_params,
                logger=self.logger,
            )

            if layer_idx == self.n_layers - 1 and self.has_odd_layer:
                # remake target X with appropriate marked entries
                solver.X = np.full(solver.X.shape, LARGE_NBR, dtype=float)
                for row, cols in home_map.items():
                    for col in cols:
                        solver.X[row, col] = np.nan  # to fill in

            # optimize
            solver.construction_phase()
            solver.tabu_phase(progress_bar)

            df_layer = solver.create_calendar().copy()

            if layer_idx == self.n_layers - 1 and self.has_odd_layer:
                # NOTE: This post-processing in the odd extra layer only keeps games
                # that match the edges, which is needed even if the solver is allowed
                # to correctly only solve for the edges in the odd layer
                allowed_pairs = {
                    (home_idx, away_idx)
                    for home_idx, opponents in home_map.items()
                    for away_idx in opponents
                }

                home_series = df_layer[col_home].map(self._team_idx_by_name)
                away_series = df_layer[col_away].map(self._team_idx_by_name)
                pair_index = pd.MultiIndex.from_arrays([home_series, away_series])

                mask_allowed = pair_index.isin(allowed_pairs)
                df_layer = df_layer[mask_allowed].reset_index(drop=True)

            df_layer["_layer"] = layer_idx
            frames.append(df_layer)

            # track occupied slots for subsequent layers
            scheduled_games = df_layer.dropna(subset=[col_date])
            for slot_date, home_team, away_team in scheduled_games[
                [col_date, col_home, col_away]
            ].itertuples(index=False, name=None):
                slot = self._slot_by_date[pd.Timestamp(slot_date)]

                home_idx = self._team_idx_by_name[home_team]
                played_slots_by_team[home_idx].add(int(slot))
                used_home_slots[home_idx].add(int(slot))

                unused_home_slots[home_idx].discard(self.input.sets["slots"][slot])

                away_idx = self._team_idx_by_name[away_team]
                played_slots_by_team[away_idx].add(int(slot))

            # TODO: Costs are underestimated / wrongly concatenated
            #   e.g. penalties for unscheduled games are not counted across layers
            #   see also commented out bit in app.py where cost of unfeasible schedules is removed
            #   -> change cost plotting so it plots the different layers separately (cf. below)
            if solver.list_full_costs:
                # NOTE: Each layer solves a 2RR subproblem, and cross-layer consistency is
                # enforced through updated availability/forbidden sets - cost traces are concatenated
                # because they represent sequential optimization work across layers and can be
                # plotted just like the default 2RR mode
                list_full_costs.extend(solver.list_full_costs)

            # NOTE: Bring the progress from out of the tabu phase to the layered phase,
            # although this will give only slight bumps in progress
            if progress_bar is not None:
                progress_bar.progress((layer_idx + 1) / self.n_layers)

        if frames:
            calendar = (
                pd.concat(frames, ignore_index=True)
                .sort_values(by=[col_date, "_layer"], na_position="last")
                .drop(columns=["_layer"])
            )
        else:
            calendar = pd.DataFrame(columns=OUTPUT_COLS)

        return LayeredPlannerResult(
            calendar=calendar,
            list_full_costs=list_full_costs,
            used_home_slots=used_home_slots,
        )

    def _prepare_layer_input(self, played_slots_by_team: dict[int, set[int]]) -> Any:
        """Prepares the input for a specific layer of the schedule, considering blocked slots and home/forbidden sets."""
        blocked_slots = self._compute_blocked_slots(played_slots_by_team)

        sets_home, sets_forbidden = {}, {}
        for team_idx in self.teams:
            blocked = blocked_slots[team_idx]

            sets_home[team_idx] = np.array(
                sorted(set(self.input.sets["home"][team_idx]) - blocked)
            )
            sets_forbidden[team_idx] = np.array(
                sorted(set(self.input.sets["forbidden"][team_idx]) | blocked)
            )

        return SimpleNamespace(
            core=self.input.core,
            locations=self.input.locations,
            sets={
                "teams": dict(self.input.sets["teams"]),
                "slots": dict(self.input.sets["slots"]),
                # teams/slots are stable across layers, only home/forbidden differ
                "home": sets_home,
                "forbidden": sets_forbidden,
            },
            parsed=True,
        )

    def _compute_blocked_slots(
        self, played_slots_by_team: dict[int, set[int]]
    ) -> dict[int, set[int]]:
        """Computes blocked slots for each team based on already played slots and the 'r_max' parameter."""
        blocked_offsets = tuple(range(-(self.params.r_max - 2), self.params.r_max - 1))

        blocked_slots = {}
        for team_idx, slots in played_slots_by_team.items():
            blocked = {slot + delta for slot in slots for delta in blocked_offsets}
            blocked_slots[team_idx] = blocked

        return blocked_slots
