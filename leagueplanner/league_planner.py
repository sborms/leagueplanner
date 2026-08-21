import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .constants import MAX_ALLOWED_REST_DAYS, OUTPUT_COLS
from .input_parser import InputParser
from .layered_planner import LayeredPlanner
from .params import PlannerParams
from .solver import Solver
from .utils import get_feasible_home_slots, get_homeless_teams

# NOTE: These are common reasons why a game remains unscheduled
# No (or too little) home availabilities
# No away availabilities on home days
# Some teams have home games on same day

# TODO: Display more flexible Python versions on PyPI (>= 3.13)
# TODO: Add example how to optimize for a pure 2RR setup (e.g. 10 rounds, 6 teams, 3 games per round)
# TODO: Allow more input flexibility (e.g. providing raw array input instead of an Excel)
# TODO: Allow starting from an existing calendar + add parameter to fix certain dates
#   (e.g. when adding a team to the league) ~ completing partially filled in schedule
# TODO: Auto-tweak input if schedule not fully completed (e.g. decrease date window for problematic teams)


class LeaguePlanner:
    """
    Generates an optimal plan/schedule/calendar for a time-relaxed double
    round-robin (2RR) league accounting for following constraints:
    - (C1) Each team plays a home game against each other team at most once.
    - (C2) Each home team its availability set (H) is respected.
    - (C3) Each away team its unavailability set (A) is respected.
    - (C4) Each team plays at most one game per time slot.
    - (C5) Each team plays at most 2 games in a period of 'r_max' time slots.
    - (C6) There are minimum 'm' time slots between two games with the same teams (pairs).

    The implementation very closely follows the tabu search based algorithm from:
    > Van Bulck, D., Goossens, D. R., & Spieksma, F. C. R. (2019).
    _Scheduling a non-professional indoor football league: a tabu search based approach._
    Annals of Operations Research, 275(2), 715-730.
    https://doi.org/10.1007/s10479-018-3013-x

    A time slot uniquely maps to a weekday (with an associated playing hour). For
    instance, if slot t is a Monday, then slot t + 3 is a Thursday, with in total 4 slots
    considered. The number of rest days is 2 in this case (Tuesday and Wednesday).
    """

    def __init__(
        self,
        input: InputParser,
        params: PlannerParams,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """
        Initializes a new instance of the LeaguePlanner class.

        See the [project README](https://github.com/sborms/leagueplanner) for
        more information about usage.

        :param input: InputParser object containing all relevant data.
        :param params: See PlannerParams for parameter details.
        :param logger: (optional) Logger instance for logging purposes.
        """
        assert input.parsed, "Input data not parsed yet!"

        # assign input data to carry along
        self.input = input
        self.params = params
        self.logger = logger

        # set all possible home slots and those feasible
        self.sets_home = input.sets["home"]
        self.sets_home_feasible = get_feasible_home_slots(self.sets_home, params.r_max)
        self.teams_without_home_slots = get_homeless_teams(self.sets_home_feasible)

        # precomputed variables
        self.games_per_opponent = params.games_per_opponent
        self.teams = self.input.sets["teams"]
        self.has_odd_layer = self.games_per_opponent % 2 == 1
        self.n_layers_full = self.games_per_opponent // 2
        self.home_maps = self._build_home_maps()

    def optimize(self, progress_bar: st.delta_generator.DeltaGenerator = None) -> list:
        """Runs the construction and tabu phases to generate an optimal schedule."""
        if self.games_per_opponent == 2:
            solver = Solver(
                input=self.input,
                params=self.params,
                logger=self.logger,
            )

            solver.construction_phase()
            solver.tabu_phase(progress_bar)

            self.list_full_costs = solver.list_full_costs
            self._used_home_slots = {
                team_idx: set(solver.X[team_idx, :]) for team_idx in self.teams
            }

            return solver.X, solver.create_calendar()
        else:
            self.logger.info(
                f"Running layered planner for {self.n_layers_full} full layer(s) and {int(self.has_odd_layer)} odd layer"
            )
            layered_planner = LayeredPlanner(
                input=self.input,
                params=self.params,
                logger=self.logger,
                # pass-along parameters
                has_odd_layer=self.has_odd_layer,
                n_layers_full=self.n_layers_full,
                home_maps=self.home_maps,
            )
            opt = layered_planner.run(progress_bar)

            self.list_full_costs = opt.list_full_costs
            self._used_home_slots = opt.used_home_slots

            return None, opt.calendar

    ##################################
    ### Calendar functionality #######
    ##################################

    def store_calendar(self, df: pd.DataFrame, file: str) -> None:
        """Stores generated calendar as an Excel file."""
        df_out = df.copy()
        df_out[OUTPUT_COLS[0]] = df_out[OUTPUT_COLS[0]].dt.strftime("%Y-%m-%d")

        df_out[OUTPUT_COLS].to_excel(file, index=False)

    def validate_calendar(
        self,
        df: pd.DataFrame,
        fl_net_rest_days: bool = False,
        cost: float = None,
    ) -> dict:
        """
        Gathers a dictionary with validation data about the generated schedule.

        :param df: Generated schedule.
        :param fl_net_rest_days: If True, returns the adjusted rest days by not counting team unavailabilities as a rest day.
        :param cost: (optional) Cost of the generated schedule. If not provided, computed from the list of full costs.
        """
        teams = self.teams
        teams_without_home_slots_names = [
            teams[team_idx] for team_idx in self.teams_without_home_slots
        ]

        d_val = {}

        # grab some general statistics first
        d_val["teams"] = len(teams)
        d_val["games"] = len(df)
        d_val["unscheduled"] = sum(df[OUTPUT_COLS[0]].isna())

        if cost is None or not hasattr(self, "list_full_costs"):
            cost = min(self.list_full_costs) if self.list_full_costs else 0
        d_val["cost"] = cost

        # overview of total number of home slots less than needed (per-team basis)
        d_req_home_games = self._compute_home_targets(teams, self.home_maps)

        n_home_slots_short = sum(
            [
                max(d_req_home_games[team_idx] - len(v), 0)
                for team_idx, v in self.sets_home_feasible.items()
            ]
        )
        d_val["missing_home_slots"] = n_home_slots_short

        # overview of number of games between two teams
        df["pairs"] = df.apply(
            lambda row: tuple(sorted([row[OUTPUT_COLS[3]], row[OUTPUT_COLS[4]]])),
            axis=1,
        )

        d_val["pairs"] = df["pairs"].value_counts()

        # overview of days between games per pair of teams
        df["days_diff"] = df.groupby("pairs")[OUTPUT_COLS[0]].diff().dt.days
        df_days_diff_pairs = (
            df[["pairs", "days_diff"]]
            .dropna()
            .sort_values("days_diff")
            .reset_index(drop=True)
        )

        d_val["min_gap_pairs"] = df_days_diff_pairs["days_diff"].min()
        d_val["max_gap_pairs"] = df_days_diff_pairs["days_diff"].max()

        # overview of (adjusted) rest days in matrix form
        df_rest_days = self.make_df_rest_days(df, net=fl_net_rest_days)
        d_val["df_rest_days"] = df_rest_days

        # overview of unused home slots
        d_val["df_unused_home_slots"] = self.make_df_unused_home_slots()

        # overview of schedules by team
        df_schedules_by_team = self.make_df_schedules_by_team(df)
        d_val["df_schedules_by_team"] = df_schedules_by_team

        # overview of regular max rest days (not adjusted!)
        series_rest_days = df_schedules_by_team["n_rest_days"].fillna(0)
        d_val["max_rest_days"] = series_rest_days.max()

        series_rest_days_excessive = series_rest_days[
            series_rest_days > MAX_ALLOWED_REST_DAYS
        ]
        d_val["n_high_rest_days_all"] = series_rest_days_excessive.count()
        d_val["n_high_rest_days_rel"] = series_rest_days_excessive[
            ~series_rest_days_excessive.index.isin(teams_without_home_slots_names)
        ].count()

        return d_val

    def make_df_rest_days(self, df: pd.DataFrame, net: bool = False) -> pd.DataFrame:
        """
        Forms a matrix of teams vs. number of rest days for given input schedule.

        :param df: DataFrame with generated schedule.
        :param net: If True, returns the adjusted rest days by not counting team unavailabilities as a rest day.
        """
        col_date = OUTPUT_COLS[0]
        col_home = OUTPUT_COLS[3]
        col_away = OUTPUT_COLS[4]
        col_team = "Team"

        df_teams = pd.concat(
            [
                df[[col_date, col_home]].rename(columns={col_home: col_team}),
                df[[col_date, col_away]].rename(columns={col_away: col_team}),
            ]
        ).sort_values([col_team, col_date])

        df_teams[col_date] = pd.to_datetime(df_teams[col_date])
        df_teams["n_rest_days"] = (
            df_teams.groupby(col_team)[col_date].diff().dt.days - 1
        )

        if net:
            col_out = "n_rest_days_net"
            df_teams = self._compute_rest_days_net(df_teams, col_team, col_date)
        else:
            col_out = "n_rest_days"

        df_out = df_teams.groupby(col_team)[col_out].value_counts().unstack().fillna(0)
        df_out = pd.concat(
            [df_out, pd.DataFrame(df_out.sum(axis=0), columns=["TOTAL"]).transpose()],
            axis=0,
        )

        return df_out

    def make_df_schedules_by_team(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorders the schedules input by team. Assumes it is already sorted by date and hour."""
        col_date = OUTPUT_COLS[0]
        col_home = OUTPUT_COLS[-2]
        col_away = OUTPUT_COLS[-1]
        col_team = "Team"

        # clean up some irrelevant columns from validation process first
        df.drop(columns=["pairs", "days_diff"], inplace=True)

        teams = pd.unique(df[[col_home, col_away]].values.ravel("K"))

        list_sch = []
        for team in teams:
            df_team = df[(df[col_home] == team) | (df[col_away] == team)].copy()
            df_team[col_team] = team
            list_sch.append(df_team)

        df_by_team = pd.concat(list_sch, ignore_index=True)

        # include rest days columns
        df_by_team["n_rest_days"] = (
            df_by_team.groupby(col_team)[col_date].diff().dt.days - 1
        )
        df_by_team = self._compute_rest_days_net(df_by_team, col_team, col_date)

        # sort by team and date
        df_by_team = df_by_team.sort_values(by=[col_team, col_date])

        # fix output format
        df_by_team = df_by_team.set_index(col_team)
        df_by_team[col_date] = df_by_team[col_date].dt.strftime("%d/%m/%Y")

        return df_by_team

    def make_df_unused_home_slots(self) -> pd.DataFrame:
        """Forms DataFrame with teams and their unused home slots."""
        if not hasattr(self, "_used_home_slots"):
            raise ValueError("Used home slots not computed yet. Run optimize() first.")

        col_date = OUTPUT_COLS[0]
        col_team = "Team"

        list_unused = []
        for team_idx, team_name in self.teams.items():
            unused_home_slots = sorted(
                [
                    self.input.sets["slots"][s]
                    for s in set(self.sets_home[team_idx]).difference(
                        self._used_home_slots[team_idx]
                    )
                ]
            )

            df_unused = pd.DataFrame({"unused": unused_home_slots})
            if len(df_unused) == 0:
                continue
            df_unused[col_team] = team_name
            list_unused.append(df_unused)

        if not list_unused:
            return pd.DataFrame(columns=[col_date])

        df_unused_all = pd.concat(list_unused)[[col_team, "unused"]].set_index(col_team)
        df_unused_all = df_unused_all.rename(columns={"unused": col_date})
        df_unused_all = df_unused_all.sort_values([col_team, OUTPUT_COLS[0]])
        df_unused_all[col_date] = df_unused_all[col_date].dt.strftime("%d/%m/%Y")

        return df_unused_all

    ##################################
    ### Plotting functionality #######
    ##################################

    def plot_minimum_costs(
        self, list_full_costs: list, title_suffix: str = "", path: str = None
    ) -> None:
        """
        Plots evolution of running minimum cost during tabu phase.

        :param list_full_costs: List of cost per iteration.
        :param title_suffix: Suffix to add to the title of the plot.
        :param path: Path to save the plot as an image (if not None).
        """
        list_running_minimum_cost = [
            min(list_full_costs[: i + 1]) for i in range(len(list_full_costs))
        ]

        # create plot
        plt.figure(figsize=(10, 6))
        plt.plot(list_running_minimum_cost)
        plt.title(
            f"Evolution minimum cost{(' - ' + title_suffix) if title_suffix else ''}"
        )
        plt.xlabel("Iteration")
        plt.tight_layout()

        # show or save plot
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

        plt.close()

    def plot_rest_days(
        self,
        series: pd.Series,
        clips: tuple = (3, 20),
        title_suffix: str = "",
        path: str = None,
    ) -> None:
        """
        Plots distribution of rest days between games.

        :param series: Series with number of rest days as index.
        :param clips: Tuple with lower and upper bound for clipping the series.
        :param title_suffix: Suffix to add to the title of the plot.
        :param path: Path to save the plot as an image (if not None).
        """
        series_ = series.copy()

        # clip series to a lower and upper bound
        if clips:
            lower, upper = clips
            l_name, u_name = f"<={lower}", f">={upper}"

            series_.index = series_.index.astype(int)

            bot = series_[series_.index <= lower].sum()
            top = series_[series_.index >= upper].sum()

            series_ = series_[(series_.index > lower) & (series_.index < upper)]
            series_.loc[lower], series_.loc[upper] = bot, top

            series_.rename(index={lower: l_name, upper: u_name}, inplace=True)

            index_no_lb = [idx for idx in series_.index if idx not in [l_name, u_name]]
            series_ = series_.reindex([l_name] + index_no_lb + [u_name])

            colors = [
                "skyblue" if (idx != l_name and idx != u_name) else "orange"
                for idx in series_.index
            ]

        # create plot
        plt.figure(figsize=(10, 6))
        series_.plot(kind="bar", color=colors if clips else "skyblue")
        plt.title(
            f"Distribution of rest days between games{(' - ' + title_suffix) if title_suffix else ''}"
        )
        plt.xlabel("Number of rest days")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # show or save plot
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

        plt.close()

    ##################################
    ### Class utils ##################
    ##################################

    def _get_df_forbidden(self) -> pd.DataFrame:
        """Creates DataFrame with forbidden time slots for each team."""
        if not hasattr(self, "df_forbidden"):
            col_date = OUTPUT_COLS[0]
            col_team = "Team"

            data = []
            for team_idx, time_slots in self.input.sets["forbidden"].items():
                team_name = self.teams[team_idx]
                for slot in time_slots:
                    date = self.input.sets["slots"][slot]
                    data.append({col_team: team_name, col_date: date})

            df_forbidden = pd.DataFrame(data)
            df_forbidden[col_date] = pd.to_datetime(df_forbidden[col_date])

            self.df_forbidden = df_forbidden

        return self.df_forbidden

    def _compute_rest_days_net(
        self, df: pd.DataFrame, col_team, col_date
    ) -> pd.DataFrame:
        """Computes net rest days for each team in the DataFrame."""
        df_forbidden = self._get_df_forbidden()

        def count_unavailable_days(
            team, start_date: pd.Timestamp, end_date: pd.Timestamp
        ) -> int:
            """Counts number of unavailable days for a team within interval."""
            mask = (
                (df_forbidden[col_team] == team)
                & (df_forbidden[col_date] > start_date)
                & (df_forbidden[col_date] < end_date)
            )
            return df_forbidden[mask].shape[0]

        df["n_rest_days_net"] = df.apply(
            lambda row: (
                row["n_rest_days"]
                - count_unavailable_days(
                    row[col_team],
                    # original start date
                    row[col_date] - pd.Timedelta(days=row["n_rest_days"] + 1),
                    row[col_date],
                )
                if pd.notnull(row["n_rest_days"])
                else None
            ),
            axis=1,
        )

        return df

    def _build_home_maps(self) -> list[dict[int, set[int]]]:
        """Builds a list of home-opponent maps per 2RR layer and one extra layer if odd round."""
        opponents_all = {
            team_idx: set(self.teams.keys()) - {team_idx} for team_idx in self.teams
        }

        home_maps = [
            {team_idx: set(opponents) for team_idx, opponents in opponents_all.items()}
            for _ in range(self.n_layers_full)
        ]

        if self.has_odd_layer:
            home_maps.append(self._build_extra_home_edges())

        return home_maps

    def _build_extra_home_edges(self) -> dict[int, set[int]]:
        """Builds a home-opponent map for the extra layer in case of an odd number of games per opponent."""
        edges = {team_idx: set() for team_idx in self.teams}

        # get one directed home edge for each pair
        # NOTE: This is not really optimized based on slot availability
        ordered_teams = sorted(self.teams.keys())
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
