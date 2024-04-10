"""Optimizer for the chore assignment problem."""

import itertools

import pandas as pd
from ortools.sat.python import cp_model

from src import config


def _parse_chore_assignment_solution(
    chores: list[tuple], people: list[str], num_periods: int, assignments: dict
) -> pd.DataFrame:
    """
    Parse the solution of the chore assignment as a DataFrame.

    Args:
        chores: List of chores to be assigned.
        people: List of people to assign chores to.
        num_periods: Number of periods to optimize over.
        assignments: Dictionary with the assignments of chores to people.

    Returns:
        A DataFrame with the optimized assignment of chores to people.
    """
    periods = [str(t + 1) for t in range(num_periods)]
    chores = [c[0] for c in chores]
    df = pd.DataFrame("", index=chores, columns=periods)
    for c in range(len(chores)):
        for t in range(num_periods):
            for p in range(len(people)):
                if assignments[(p, c, t)]:
                    df.loc[chores[c], str(t + 1)] = people[p]
    return df


class ChoresPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(
        self,
        chores: list[tuple],
        people: list[str],
        num_periods: int,
        assignments: dict,
    ) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._chores = chores
        self._people = people
        self._assignments = assignments
        self._num_chores = len(chores)
        self._num_people = len(people)
        self._num_periods = num_periods
        self._solution_count = 0

    def on_solution_callback(self) -> None:
        self._solution_count += 1
        obj = self.ObjectiveValue()
        print(f"Solution {self._solution_count} (obj. value {obj}):")

        assignments = {k: self.Value(v) for k, v in self._assignments.items()}
        df = _parse_chore_assignment_solution(
            chores=self._chores,
            people=self._people,
            num_periods=self._num_periods,
            assignments=assignments,
        )

        print(df.to_string(col_space=2, justify="right") + "\n")


def _add_constraints_to_cp_model(
    model: cp_model.CpModel,
    chores: list[tuple],
    num_people: int,
    num_periods: int,
    assignments: dict,
) -> None:
    """
    Add constraints to the chore assignment model.

    Args:
        model: The constraint programming model.
        chores: List of chores to be assigned.
        num_people: Number of people to assign chores to.
        num_periods: Number of periods to optimize over.
        assignments: Dictionary with the assignments of chores to people.
    """
    num_chores = len(chores)
    chore_intervals = [c[1] for c in chores]

    # Criterion 1: Each chore must be performed at regular intervals
    for c in range(num_chores):
        for t in range(num_periods - chore_intervals[c]):
            chore_performed_at_t = sum(
                assignments[(p, c, t)] for p in range(num_people)
            )
            chore_performed_at_t_plus_interval = sum(
                assignments[(p, c, t + chore_intervals[c])]
                for p in range(num_people)
            )
            model.Add(
                chore_performed_at_t == chore_performed_at_t_plus_interval
            )

    # Criterion 2: Each chore must be performed at least N times and at most
    # N + 1 times, where N is the number of periods divided by the interval of
    # the chore
    for c in range(num_chores):
        n_times_chore_performed = sum(
            assignments[(p, c, t)]
            for p in range(num_people)
            for t in range(num_periods)
        )
        chore_times_to_perform_lower_bound = num_periods // chore_intervals[c]
        model.Add(n_times_chore_performed >= chore_times_to_perform_lower_bound)
        model.Add(
            n_times_chore_performed <= chore_times_to_perform_lower_bound + 1
        )

        # Special case for chores that can be performed at most once
        if chore_intervals[c] >= num_periods:
            model.Add(n_times_chore_performed == 1)

    # Criterion 3: In a given period, the scheduled chores must each be
    # performed by exactly one person
    for c in range(num_chores):
        for t in range(num_periods):
            model.add_at_most_one(
                assignments[(p, c, t)] for p in range(num_people)
            )

    # Criterion 4: Each chore must be assigned approximately the same number of
    # times to each person
    for c in range(num_chores):
        for p in range(num_people):
            n_times_chore_assigned_to_person = sum(
                assignments[(p, c, t)] for t in range(num_periods)
            )
            n_times_minimum_per_person = num_periods // (
                num_people * chore_intervals[c]
            )
            model.Add(
                n_times_chore_assigned_to_person >= n_times_minimum_per_person
            )

    # Criterion 5: Never assign the same to chore to the same person in
    # consecutive intervals
    for c in range(num_chores):
        for p in range(num_people):
            for t in range(num_periods - chore_intervals[c]):
                model.add_at_most_one(
                    assignments[(p, c, t)],
                    assignments[(p, c, t + chore_intervals[c])],
                )

    # Symmetry breaking: assign the first chore to the first person in period 0
    model.Add(assignments[(0, 0, 0)] == 1)


def _add_objective_to_cp_model(
    model: cp_model.CpModel,
    chores: list[tuple],
    num_people: int,
    num_periods: int,
    assignments: dict,
) -> None:
    """
    Add the objective function to the chore assignment model.

    Args:
        model: The constraint programming model.
        chores: List of chores to be assigned.
        num_people: Number of people to assign chores to.
        num_periods: Number of periods to optimize over.
        assignments: Dictionary with the assignments of chores to people.
    """
    chore_intervals = [c[1] for c in chores]
    chore_workloads = [c[2] for c in chores]
    num_chores = len(chores)

    # Objective 1: Spread the workload as evenly as possible across periods by
    # minimizing the difference between the workloads of each period and the
    # average workload
    total_workload = sum(
        (num_periods / chore_intervals[c]) * chore_workloads[c]
        for c in range(num_chores)
    )
    avg_workload_periods = round(total_workload / (num_people * num_periods))

    workload_diff_across_periods = []
    for p in range(num_people):
        for t in range(num_periods):
            workload = sum(
                (assignments[(p, c, t)] * chore_workloads[c])
                for c in range(num_chores)
            )
            workload_diff = model.NewIntVar(
                0, 999, f"workload_diff_person_{p}_period_{t}"
            )
            model.add_abs_equality(
                workload_diff, workload - avg_workload_periods
            )
            workload_diff_across_periods.append(workload_diff)

    # Objective 2: Spread the workload as evenly as possible across people by
    # minimizing the difference between the total workloads of each person and
    # the average workload of each person
    avg_workload_person = round(total_workload / num_people)

    workload_diff_persons = []
    for p in range(num_people):
        workload = sum(
            (assignments[(p, c, t)] * chore_workloads[c])
            for c in range(num_chores)
            for t in range(num_periods)
        )
        workload_diff = model.NewIntVar(0, 999, f"workload_diff_person_{p}")
        model.add_abs_equality(workload_diff, workload - avg_workload_person)
        workload_diff_persons.append(workload_diff)

    model.minimize(
        sum(workload_diff_across_periods) + sum(workload_diff_persons)
    )


def optimize_chore_assignment(
    chores: list[tuple], people: list[str], num_periods: int
) -> pd.DataFrame:
    """
    Optimize the assignment of chores to people over a number of periods.

    Args:
        chores: List of chores to be assigned.
        people: List of people to assign chores to.
        num_periods: Number of periods to optimize over.

    Returns:
        A DataFrame with the optimized assignment of chores to people.
    """
    num_people = len(people)
    num_chores = len(chores)

    model = cp_model.CpModel()

    # Create the boolean tensor of who is doing what chore in what period
    combinations = itertools.product(
        range(num_people), range(num_chores), range(num_periods)
    )
    assignments = {
        (p, c, t): model.NewBoolVar(f"person_{p}_chore_{c}_period_{t}")
        for p, c, t in combinations
    }

    # Add constraints to the model
    _add_constraints_to_cp_model(
        model=model,
        chores=chores,
        num_people=num_people,
        num_periods=num_periods,
        assignments=assignments,
    )

    # Add the objective function to the model
    _add_objective_to_cp_model(
        model=model,
        chores=chores,
        num_people=num_people,
        num_periods=num_periods,
        assignments=assignments,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.max_time_in_seconds

    solution_printer = ChoresPartialSolutionPrinter(
        chores, people, num_periods, assignments
    )
    status = solver.solve(model, solution_printer)

    print(f"Status: {solver.StatusName(status)}")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _parse_chore_assignment_solution(
            chores=chores,
            people=people,
            num_periods=num_periods,
            assignments={k: solver.Value(v) for k, v in assignments.items()},
        )
    else:
        raise ValueError("No feasible solution found within the time limit.")
