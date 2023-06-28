# -*- coding: utf-8 -*-
"""
BROT (Bilevel Routing on networks with Optimal Transport) -- https://github.com/aleable/BROT

Contributors:
    Alessandro Lonardi
    Caterina De Bacco
"""

import numpy as np
from scipy.sparse import csr_matrix


def assert_steps(itcheckstart: int, totsteps: int) -> None:
    """
    Raise exception if number of steps for each method is lower than threshold in convergence check.

    :param itcheckstart: starting step for convergence check
    :param totsteps: total time steps
    :return: None
    """

    if itcheckstart < totsteps:
        pass
    else:
        raise ValueError("[ERROR] IT where conv is checked is too large")

    return None


def assert_topol(topol: str) -> None:
    """
    Raise exception if topol type is not valid.

    :param topol: network topology
    :return: None
    """

    if topol in ["euroroads", "synthetic", "disk", "lattice"]:
        pass
    else:
        raise ValueError("[ERROR] topol type not valid")

    return None


def assert_mass_balance(S: np.ndarray) -> None:
    """
    Raise Exception if mass does not sum to zero.

    :param S: forcing
    :return: None
    """

    eps = 1.0e-5

    checks = list()
    for i in range(S.shape[1]):
        if abs(sum(S[:, i])) < eps:
            checks.append(True)

    if all(c is True for c in checks):
        pass
    else:
        raise ValueError("[ERROR] mass is not balanced")

    return None


def assert_mass_conservation(B: csr_matrix, F: np.ndarray, S: np.ndarray) -> None:
    """
    Raise Exception if mass is not balanced in OT/GD updates.

    :param B: incidence matrix
    :param F: fluxes
    :param S: forcing
    :return: None
    """

    eps = 1.0e-5
    check = np.sum(abs(B * F - S))
    if check < eps:
        pass
    else:
        raise ValueError(f"[ERROR] mass is not conserved: {check} > {eps}")

    return None


def assert_positivity_w(w: np.ndarray) -> None:
    """
    Raise Exception if weights become negative during GD updates.

    :param w: weights
    :return: None
    """

    checks = list(w > 0)
    if all(checks):
        pass
    else:
        raise ValueError(f"[ERROR] negative weights in GD update")

    return None


def assert_positivity_eps(eps: float) -> None:
    """
    Raise Exception if epsilon is negative or zero.

    :param eps: weights
    :return: None
    """

    if eps >= 0:
        pass
    else:
        raise ValueError(f"[ERROR] negative epsilon")

    return None


def assert_projection(proj: str) -> None:
    """
    Raise Exception if projection method is not valid.

    :param proj: projection method
    :return: None
    """

    if proj in ["momentum", "clipping"]:
        pass
    else:
        raise ValueError("[ERROR] projection method not valid")

    return None


def print_main(verboseflag: bool, whichprint: int, data: dict) -> None:
    """
    Print BROT metadata.

    :param verboseflag: verbose flag
    :param whichprint: print option flag
    :param data: data to print
    :return: None
    """

    if verboseflag and (whichprint == 0):
        print("\033[91m" + 60 * "=" + f"| START" + "\033[0m")

    if verboseflag and (whichprint == 1):
        print("\033[91m" + 60 * "=" + f"| STOP" + "\033[0m")

    if verboseflag and (whichprint == 2):
        print(f"elapsed time = {data['time']} [s]")

    return None


def print_brot(
    verboseflag: bool,
    whichprint: int,
    step: int,
    c: float,
    dc: float,
    TIME_STEP_EXP_VERB: int,
) -> None:
    """
    Print BROT metadata.

    :param verboseflag: verbose flag
    :param whichprint: print option flag
    :param step: time step
    :param c: cost
    :param dc: cost difference
    :param TIME_STEP_EXP_VERB: time frequency for serialization of variables
    :return: None
    """

    if verboseflag and (whichprint == 0) and (step % TIME_STEP_EXP_VERB == 0):
        print("\033[91m" + 30 * "=" + f"> STEP = {step}" + "\033[0m")

    if verboseflag and (whichprint == 1) and (step % TIME_STEP_EXP_VERB == 0):
        print(f"C = {round(c, 8)} - DC = {round(dc, 8)}")

    if verboseflag and (whichprint == 2) and (step % TIME_STEP_EXP_VERB == 0):
        print("\033[91m" + 30 * "=" + f"> STEP = {step}" + "\033[0m")

    return None


def print_dyn(verboseflag: bool, step: int, TIME_STEP_EXP_VERB: int) -> None:
    """
    Print OT dynamics metadata.

    :param verboseflag: verbose flag
    :param step: time step
    :param TIME_STEP_EXP_VERB: time frequency for serialization of variables
    :return: None
    """

    if verboseflag and (step % TIME_STEP_EXP_VERB == 0):
        print("\033[94m" + 30 * "=" + "> OT" + "\033[0m")

    return None


def print_descent(verboseflag: bool, step: int, TIME_STEP_EXP_VERB: int) -> None:
    """
    Print GD metadata

    :param verboseflag: verbose flag
    :param step: time step
    :param TIME_STEP_EXP_VERB: time frequency for serialization of variables
    :return: None
    """

    if verboseflag and (step % TIME_STEP_EXP_VERB == 0):
        print("\033[92m" + 30 * "=" + "> GD" + "\033[0m")

    return None
