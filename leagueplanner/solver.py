import bisect
import logging
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st

from leagueplanner.utils import get_feasible_home_slots, get_homeless_teams

from .constants import LARGE_NBR, MAX_ALLOWED_REST_DAYS, OUTPUT_COLS
from .input_parser import InputParser
from .params import PlannerParams
from .perturbation import Perturbation
from .transportation_problem_solver import TransportationProblemSolver as TPS


class Solver:
    """Exposes the core algorithm to solve the scheduling problem."""

    def __init__(
        self,
        input: InputParser,
        params: PlannerParams,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """See LeaguePlanner for details."""
        # assign input data to carry along
        self.input = input
        self.params = params
        self.logger = logger

        # algorithm parameters
        self.tabu_length = params.tabu_length
        self.perturbation_length = params.perturbation_length
        self.n_iterations = params.n_iterations
        self.cost_excessive_rest_days = params.cost_excessive_rest_days

        # precomputed variables
        self.teams = self.input.sets["teams"]
        self.n_teams = len(self.teams)
        self._rest_days_buf = np.empty((self.n_teams, 2 * self.n_teams))

        # initialize target matrix with teams & slots
        X = np.eye(self.n_teams) * LARGE_NBR  # diagonal is to be ignored
        self.X = np.where(X == 0, np.nan, X)

        # set all possible home slots and those feasible (cf. LeaguePlanner)
        self.sets_home = input.sets["home"]
        self.sets_home_feasible = get_feasible_home_slots(self.sets_home, params.r_max)
        self.teams_without_home_slots = get_homeless_teams(self.sets_home_feasible)

        # initialize transportation object
        self.tps = TPS(
            sets_forbidden=input.sets["forbidden"],
            sets_home=self.sets_home_feasible,
            m=params.m,
            p=params.p,
            r_max=params.r_max,
            penalties=params.penalties,
        )

        # initialize perturbation object
        self.perturbation = Perturbation(alpha=params.alpha, beta=params.beta)

        # track top X matrices with lowest cost as a list of dicts {"cost": float, "X": np.ndarray}
        self.top_X = []
        self._top_X_costs = []  # parallel list for fast bisect insertion

        # initialize list with costs per home team
        self.list_home_costs = [None] * self.n_teams

    def construction_phase(self) -> None:
        """Generates initial (possibly incomplete) schedule and assigns it to self.X."""
        self.logger.info("Phase 1: Construction")

        # method 1
        # repeatedly select team with smallest number of available home slots
        X1 = self.X.copy()
        list_home_costs1 = self.list_home_costs.copy()
        d_spots1 = self._update_dict_available_spots(method=1)  # initialize dict

        for _ in range(self.n_teams):
            team_idx = list(d_spots1)[0]  # pick team

            # solve transportation problem for home team in current schedule X1
            X1, total_cost = self.tps.solve(X1, team_idx)
            list_home_costs1[team_idx] = total_cost

            # update available spots
            d_spots1 = self._update_dict_available_spots(1, X1, d_spots1, team_idx)

        cost1 = sum(list_home_costs1)
        cost1 += self._count_excessive_rest_days(X1) * self.cost_excessive_rest_days
        self.logger.info(f"Initialized schedule using method 1 with cost {cost1}")

        # method 2
        # repeatedly select team with smallest number of possible games
        X2 = self.X.copy()
        list_home_costs2 = self.list_home_costs.copy()
        d_spots2 = self._update_dict_available_spots(method=2, X=X2)  # initialize dict

        for _ in range(self.n_teams):
            team_idx = list(d_spots2)[0]  # pick team

            # solve transportation problem for home team in current schedule X2
            X2, total_cost = self.tps.solve(X2, team_idx)
            list_home_costs2[team_idx] = total_cost

            # update available spots
            d_spots2 = self._update_dict_available_spots(2, X2, d_spots2, team_idx)

        cost2 = sum(list_home_costs2)
        cost2 += self._count_excessive_rest_days(X2) * self.cost_excessive_rest_days
        self.logger.info(f"Initialized schedule using method 2 with cost {cost2}")

        # NOTE: Home costs don't take into account later assigned games but the tabu
        # phase will account for it - however there is a slight risk that a new best
        # between the downward biased starting point and the actual cost is missed;
        # there is always a slight delay between the actual cost and the reported cost

        # pick best method to set schedule after construction phase
        self.logger.info(f"Comparing costs {cost1} vs. {cost2}")
        if cost1 < cost2:
            self.logger.info("Initialization method 1 is best")
            self.X, self.list_home_costs = X1, list_home_costs1
        else:
            self.logger.info("Initialization method 2 is best")
            self.X, self.list_home_costs = X2, list_home_costs2

        # initialize list with full costs
        self.list_full_costs = [cost1 if cost1 < cost2 else cost2]

    def tabu_phase(
        self, progress_bar: st.delta_generator.DeltaGenerator = None
    ) -> None:
        """
        Solves transportation problem to (re)schedule all home games of
        a non-tabu team (= not recently chosen), for a certain number of
        iterations or until the full cost reaches zero. Every new optimal
        schedule is added to self.X.

        Note that this implementation nowhere enforces a minimal cost change
        before allowed to continue to the next iteration.

        :param progress_bar: A progress bar object, e.g., streamlit.progress(0.0).
        """
        self.logger.info("Phase 2: Tabu & perturbation")

        X = self.X.copy()  # get current schedule

        list_home_costs = self.list_home_costs.copy()

        full_cost_min = self.list_full_costs[-1]
        self.logger.info(f"Tabu phase starts with cost {full_cost_min}")

        list_tabu = deque()
        list_nontabu = list(self.input.sets["teams"].keys())

        it = 0
        while it < self.n_iterations and self.list_full_costs[-1] > 0:
            it += 1
            if progress_bar is not None and it % 10 == 0:
                progress_bar.progress(it / self.n_iterations)

            # check if current schedule needs to be perturbated
            if it % self.perturbation_length == 0:
                # perturbate if no better solution found for a while, else keep going
                if (
                    self.list_full_costs[-1]
                    >= self.list_full_costs[-self.perturbation_length]
                ):
                    n_unsched_pre = np.sum(np.isnan(X), axis=1)
                    self.logger.info(f"Perturbating schedule at iteration {it}")
                    self.perturbation.perturbate(X)

                    # adjust costs based on dropped games from perturbation
                    n_unsched_pos = np.sum(np.isnan(X), axis=1)
                    list_home_costs += (n_unsched_pos - n_unsched_pre) * self.tps.p

            # recover team that has been tabu_length iterations in tabu list
            if it > self.tabu_length:
                team_nontabu = list_tabu.popleft()
                list_nontabu.append(team_nontabu)

            # randomly choose non-tabu team
            team_idx = np.random.choice(list_nontabu)
            list_tabu.append(team_idx)
            list_nontabu.remove(team_idx)

            # reschedule home games of picked team
            X[team_idx, :] = np.nan
            X[team_idx, team_idx] = LARGE_NBR

            X, total_cost = self.tps.solve(X, team_idx)

            # update costs, first including the cost for the excessive rest days
            list_home_costs[team_idx] = total_cost
            n_excessive_rest_days = self._count_excessive_rest_days(X)

            full_cost = (
                sum(list_home_costs)
                + n_excessive_rest_days * self.cost_excessive_rest_days
            )
            self.list_full_costs.append(full_cost)

            # update top_X with current X and cost
            self._update_top_X(full_cost, X)

            # check quality
            if full_cost < full_cost_min:  # new best
                self.logger.info(
                    f"!!! New best at iteration {it:>6} -> "
                    f"{full_cost:>9.1f} < {full_cost_min:<9.1f} | "
                    f"Excessive rest days = {n_excessive_rest_days}"
                )
                full_cost_min = full_cost
                self.X = X.copy()  # update to new optimal schedule

        # set progress bar to 100% (needed in case of early termination)
        if progress_bar is not None:
            progress_bar.progress(1.0)

    ##################################
    ### Calendar functionality #######
    ##################################

    def create_calendar(self) -> pd.DataFrame:
        """Creates a calendar DataFrame from the optimal schedule in provided X."""
        X = self.X
        teams = self.teams
        core = self.input.core
        locations = self.input.locations
        set_slots = self.input.sets["slots"]

        list_team, list_oppo, list_location, list_date, list_hour = [], [], [], [], []

        for i in range(X.shape[0]):
            team = teams[i]
            for j in range(X.shape[1]):
                if i == j:
                    continue
                oppo = teams[j]

                list_team.append(team)
                list_oppo.append(oppo)
                list_location.append(locations[team])

                slot = X[i, j]
                if not pd.isna(slot):
                    list_date.append(set_slots[slot])
                    list_hour.append(core[team].loc[slot])
                else:
                    list_date.append(np.nan)
                    list_hour.append(np.nan)

        df = pd.DataFrame(
            {
                OUTPUT_COLS[0]: list_date,
                OUTPUT_COLS[1]: list_hour,
                OUTPUT_COLS[2]: list_location,
                OUTPUT_COLS[3]: list_team,
                OUTPUT_COLS[4]: list_oppo,
            }
        )

        df[OUTPUT_COLS[0]] = pd.to_datetime(df[OUTPUT_COLS[0]])
        df = df.sort_values(OUTPUT_COLS[0])

        return df

    ##################################
    ### Class utils ##################
    ##################################

    def _update_dict_available_spots(
        self,
        method: int,
        X: np.ndarray = None,
        d_spots: dict = None,
        team_idx_last: int = None,
    ) -> dict:
        """Updates available home/game spots for each team during construction phase."""
        if team_idx_last is not None:
            # drop last processed team
            d_spots.pop(team_idx_last)

        if method == 1:
            if d_spots is None:
                # initialize spots from available home time slots
                d_spots = {
                    key: len(self.sets_home[key]) for key in self.input.sets["teams"]
                }
            else:
                # subtract current scheduled away games
                d_spots = {
                    key: v - np.sum(np.isin(self.sets_home[key], X[:, key]))
                    for key, v in d_spots.items()
                }
        elif method == 2:
            if d_spots is None:
                d_spots = dict.fromkeys(self.input.sets["teams"])
            for team_idx in d_spots:
                set_home = self.sets_home[team_idx]
                opponents = [t for t in range(X.shape[0]) if t != team_idx]

                m = self.tps.create_cost_matrix(X, team_idx, set_home, opponents)

                # count number of home slots possible for each opponent
                home_option_score = np.sum(m == 0, axis=0).min()
                d_spots[team_idx] = home_option_score

        # sort by number of spots (low to high)
        d_spots = dict(sorted(d_spots.items(), key=lambda x: x[1]))

        return d_spots

    def _update_top_X(self, cost: float, X: np.ndarray, n: int = 10) -> None:
        """
        Adds the current X and cost to self.top_X if it belongs in the top 'n' lowest costs.
        Keeps self.top_X sorted by ascending cost with max length 'n'.
        """
        # skip if cost doesn't qualify for top n
        if len(self.top_X) >= n and cost >= self._top_X_costs[-1]:
            return

        entry = {"cost": cost, "X": np.copy(X)}
        idx = bisect.bisect_left(self._top_X_costs, cost)
        self.top_X.insert(idx, entry)
        self._top_X_costs.insert(idx, cost)
        if len(self.top_X) > n:
            self.top_X.pop()
            self._top_X_costs.pop()

    def _count_excessive_rest_days(self, X: np.ndarray) -> float:
        """
        Returns how often the all teams have rest days > MAX_ALLOWED_REST_DAYS for the given schedule.
        Ignores a team if it has no available home slots (= homeless).
        """
        LARGE_SENTINEL = 1e18

        n = self.n_teams

        # combine all games per team using preallocated buffer: row (home games) + column (away games)
        buf = self._rest_days_buf
        buf[:, :n] = X
        buf[:, n:] = X.T

        # mask invalid slots (NaN and LARGE_NBR diagonal), replace with finite sentinel
        buf[np.isnan(buf) | (buf == LARGE_NBR)] = LARGE_SENTINEL

        # sort each team's slots in-place
        buf.sort(axis=1)

        # count excessive rest days between consecutive valid games using direct slice subtraction
        right = buf[:, 1:]
        excessive = (right - buf[:, :-1] > MAX_ALLOWED_REST_DAYS + 1) & (
            right < LARGE_SENTINEL
        )

        # exclude teams without home slots
        if self.teams_without_home_slots:
            excessive[self.teams_without_home_slots, :] = False

        return int(excessive.sum())
