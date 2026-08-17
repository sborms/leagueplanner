import copy
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .input_parser import InputParser
from .league_planner import LeaguePlanner
from .params import PlannerParams


@dataclass
class LayeredPlannerResult:
    calendar: pd.DataFrame
    home_targets: dict[int, int]
    used_home_slots: dict[int, set[int]]
    list_full_costs: list[float]


class LayeredPlanner:
    """Runs the LeaguePlanner in a layered fashion for the non-default case of games_per_opponent != 2."""

    def __init__(
        self,
        input: InputParser,
        params: PlannerParams,
        logger: logging.Logger,
    ) -> None:
        """See LeaguePlanner for details."""

        self.input = input
        self.params = params
        self.logger = logger

        self.output_cols = LeaguePlanner.output_cols

        # precomputed variables
        self.n_teams = len(self.input.sets["teams"])  # also in LeaguePlanner()
        self.team_indices = list(self.input.sets["teams"].keys())
        self.team_idx_by_name = {
            team_name: team_idx
            for team_idx, team_name in self.input.sets["teams"].items()
        }
        self._slot_by_date = {
            pd.Timestamp(date): slot for slot, date in self.input.sets["slots"].items()
        }
        self._has_odd_round = self.params.games_per_opponent % 2 == 1
        self._blocked_offsets = tuple(
            range(-(self.params.r_max - 2), self.params.r_max - 1)
        )

    def run(
        self, progress_bar: st.delta_generator.DeltaGenerator = None
    ) -> LayeredPlannerResult:
        col_date = self.output_cols[0]
        col_home = self.output_cols[3]
        col_away = self.output_cols[4]

        # build one home-opponent map per 2RR layer and one extra layer for odd round
        home_maps = self._build_layer_home_maps()
        home_targets = self._compute_home_targets(self.input.sets["teams"], home_maps)

        played_slots_by_team = {team_idx: set() for team_idx in self.team_indices}
        used_home_slots = {team_idx: set() for team_idx in self.team_indices}

        n_layers = len(home_maps)

        list_full_costs, frames = [], []
        for layer_idx, home_map in enumerate(home_maps):
            # c_p = {k: len(v) for k, v in played_slots_by_team.items()}
            # c_h = {k: len(v) for k, v in used_home_slots.items()}
            # print(
            #     f"Running layer {layer_idx + 1}/{n_layers}\n"
            #     f"Home map:\n{home_map}\n"
            #     f"Played slots:\n{played_slots_by_team}\n"
            #     f"Played slots (counts):\n{c_p}\n"
            #     f"Used home slots:\n{used_home_slots}\n"
            #     f"Used home slots (counts):\n{c_h}"
            # )

            layer_input = self._prepare_layer_input(
                layer_idx=layer_idx,
                n_layers=n_layers,
                played_slots_by_team=played_slots_by_team,
            )

            layer_params = copy.deepcopy(self.params)
            layer_params.games_per_opponent = 2

            layer_planner = LeaguePlanner(
                input=layer_input,
                params=layer_params,
                logger=self.logger,
            )
            layer_planner.construction_phase()
            layer_planner.tabu_phase()

            df_layer = layer_planner.create_calendar().copy()

            if layer_idx == n_layers - 1 and self._has_odd_round:
                # in the odd extra layer only keep games that match the edges
                allowed_pairs = {
                    (home_idx, away_idx)
                    for home_idx, opponents in home_map.items()
                    for away_idx in opponents
                }

                home_series = df_layer[col_home].map(self.team_idx_by_name)
                away_series = df_layer[col_away].map(self.team_idx_by_name)
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

                home_idx = self.team_idx_by_name[home_team]
                played_slots_by_team[home_idx].add(int(slot))
                used_home_slots[home_idx].add(int(slot))

                away_idx = self.team_idx_by_name[away_team]
                played_slots_by_team[away_idx].add(int(slot))

            # TODO: Costs are underestimated / wrongly concatenated
            #   e.g. penalties for unscheduled games are not counted across layers
            #   see also commented out bit in app.py where cost of unfeasible schedules is removed
            if layer_planner.list_full_costs:
                # NOTE: Each layer solves a 2RR subproblem, and cross-layer consistency is
                # enforced through updated availability/forbidden sets - cost traces are concatenated
                # because they represent sequential optimization work across layers and can be
                # plotted just like the default 2RR mode
                list_full_costs.extend(layer_planner.list_full_costs)

            # NOTE: Bring the progress from out of the tabu phase to the layered phase,
            # although this will give only slight bumps in progress
            if progress_bar is not None:
                progress_bar.progress((layer_idx + 1) / n_layers)

        if frames:
            calendar = (
                pd.concat(frames, ignore_index=True)
                .sort_values(by=[col_date, "_layer"], na_position="last")
                .drop(columns=["_layer"])
            )
        else:
            calendar = pd.DataFrame(columns=self.output_cols)

        return LayeredPlannerResult(
            calendar=calendar,
            home_targets=home_targets,
            used_home_slots=used_home_slots,
            list_full_costs=list_full_costs,
        )

    def _build_layer_home_maps(self) -> list[dict[int, set[int]]]:
        """Builds a list of home-opponent maps for each layer of the schedule."""
        opponents_all = {
            team_idx: set(self.team_indices).difference({team_idx})
            for team_idx in self.team_indices
        }
        n_full_rounds = self.params.games_per_opponent // 2

        home_maps = [
            {team_idx: set(opponents) for team_idx, opponents in opponents_all.items()}
            for _ in range(n_full_rounds)
        ]

        if self._has_odd_round:
            home_maps.append(self._build_extra_home_edges())

        return home_maps

    def _build_extra_home_edges(self) -> dict[int, set[int]]:
        """Builds a home-opponent map for the extra layer in case of an odd number of games per opponent."""
        edges = {team_idx: set() for team_idx in self.team_indices}

        # get one directed home edge for each pair
        # NOTE: This is not really optimized based on slot availability
        ordered_teams = sorted(self.team_indices)
        for i_pos, i in enumerate(ordered_teams):
            for j_pos in range(i_pos + 1, len(ordered_teams)):
                j = ordered_teams[j_pos]
                if (i_pos + j_pos) % 2 == 0:
                    edges[i].add(j)
                else:
                    edges[j].add(i)

        return edges

    def _compute_home_targets(
        self, teams: dict[int, str], home_maps: list[dict[int, set[int]]]
    ) -> dict[int, int]:
        """Computes the total number of home games each team should play across all layers."""
        targets = dict.fromkeys(teams, 0)
        for home_map in home_maps:
            for team_idx, opponents in home_map.items():
                targets[team_idx] += len(opponents)

        return targets

    def _prepare_layer_input(
        self,
        layer_idx: int,
        n_layers: int,
        played_slots_by_team: dict[int, set[int]],
    ) -> Any:
        """Prepares the input for a specific layer of the schedule, considering blocked slots and home/forbidden sets."""
        blocked_slots = self._compute_blocked_slots(played_slots_by_team)

        min_needed = max(1, self.n_teams - 1)

        sets_home, sets_forbidden = {}, {}
        for team_idx in self.team_indices:
            all_home = np.array(self.input.sets["home"][team_idx], dtype=int)

            team_blocked = blocked_slots[team_idx]

            allowed_home = np.array(
                [slot for slot in all_home if slot not in team_blocked], dtype=int
            )
            if len(allowed_home) > 0:
                bucket_home = allowed_home[layer_idx::n_layers]
                sets_home[team_idx] = (
                    bucket_home if len(bucket_home) >= min_needed else allowed_home
                )
            else:
                sets_home[team_idx] = allowed_home

            forbidden = {int(v) for v in self.input.sets["forbidden"][team_idx]}
            forbidden.update(team_blocked)

            sets_forbidden[team_idx] = np.array(sorted(forbidden), dtype=int)

        return SimpleNamespace(
            parsed=True,
            sets={
                # team/slot maps are stable across layers, only home/forbidden differ
                "teams": dict(self.input.sets["teams"]),
                "slots": dict(self.input.sets["slots"]),
                "home": sets_home,
                "forbidden": sets_forbidden,
            },
            core=self.input.core,
            locations=self.input.locations,
        )

    def _compute_blocked_slots(
        self, played_slots_by_team: dict[int, set[int]]
    ) -> dict[int, set[int]]:
        """Computes blocked slots for each team based on already played slots and the 'r_max' parameter."""
        blocked_slots = {}
        for team_idx, slots in played_slots_by_team.items():
            blocked = {
                slot + delta for slot in slots for delta in self._blocked_offsets
            }
            blocked_slots[team_idx] = blocked

        return blocked_slots
