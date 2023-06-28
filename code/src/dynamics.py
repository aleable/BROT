# -*- coding: utf-8 -*-
"""
BROT (Bilevel Routing on networks with Optimal Transport) -- https://github.com/aleable/BROT

Contributors:
    Alessandro Lonardi
    Caterina De Bacco
"""

import sys

import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import spsolve

from .utils import assert_mass_conservation, print_dyn


def otdyn(
    mu: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    B: csr_matrix,
    N: int,
    E: int,
    M: int,
    theta: float,
    delta: float,
    TOT: float,
    relax: float,
    verbose: bool,
    step: int,
    GDexec: bool,
    TSEV: int,
) -> {np.ndarray, float, list, list, list, list, list}:
    """
    OT dynamics.

    :param mu: conductivities
    :param w: weights
    :param S: forcing
    :param B: incidence matrix
    :param N: number of nodes
    :param E: number of edges
    :param M: number of commodities
    :param theta: critical threshold for congestion
    :param delta: time step for OT dynamics
    :param TOT: time steps OT dynamics
    :param relax: relaxation for Laplacian inversion
    :param verbose: verbose flag for BROT metadata
    :param step: time step
    :param GDexec: flag to execute GD
    :param TSEV: time frequency for serialization of variables
    :return: conductivities, J difference, {J, Omega, mu, w, F} evolution
    """

    print_dyn(verbose, step, TSEV)

    # INITIALIZATION
    it = 0
    p = get_potential(mu, w, S, B, N, M, relax)
    J, F, Fnorm = get_ot_cost_and_flux(mu, w, p, B, M, E)

    dJ = sys.float_info.max
    Jstack = list()
    Omegastack = list()
    mustack = list()
    wstack = list()
    Fstack = list()

    if (not GDexec) and (step == 0):
        Jstack.append(J)
        mustack.append(mu)
        wstack.append(w)
        Fstack.append(F)
        Omega = get_gd_cost(Fnorm, theta)
        Omegastack.append(Omega)
        it = 1

    # UPDATE
    while it < TOT:

        mu = update_mu(mu, p, w, B, M, delta)
        p = get_potential(mu, w, S, B, N, M, relax)
        J_new, F, Fnorm = get_ot_cost_and_flux(mu, w, p, B, M, E)
        dJ = compute_diff_cost(J_new, J, delta)

        J = J_new

        Jstack.append(J)
        mustack.append(mu)
        wstack.append(w)
        Fstack.append(F)
        Omega = get_gd_cost(Fnorm, theta)
        Omegastack.append(Omega)

        assert_mass_conservation(B, F, S)

        it += 1

    return mu, dJ, Jstack, Omegastack, mustack, wstack, Fstack


def get_potential(
    mu: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    B: csr_matrix,
    N: int,
    M: int,
    relax: float,
) -> np.ndarray:
    """
    Compute potential.

    Sizes of variables:
        - |p| = NxM

    :param mu: conductivities
    :param w: weights
    :param S: forcing
    :param B: incidence matrix
    :param N: number of nodes
    :param M: number of commodities
    :param relax: elaxation for Laplacian inversion
    :return: potential
    """

    p = np.zeros((N, M))

    for i in range(M):
        Li = B * diags(mu[:, i] / w) * B.T + relax * identity(N)
        p[:, i] = spsolve(Li, S[:, i])

    return p


def get_ot_cost_and_flux(
    mu: np.ndarray, w: np.ndarray, p: np.ndarray, B: csr_matrix, M: int, E: int
) -> {float, np.ndarray, np.ndarray}:
    """
    Compute J and F.

    Sizes of variables:
        - |F| = ExM

    :param mu: conductivities
    :param w: weights
    :param p: potential
    :param B: incidence matrix
    :param M: number of commodities
    :param E: number of edges
    :return: J, fluxes, 1-norm of fluxes
    """

    F = np.zeros((E, M))

    for i in range(M):
        F[:, i] = csr_matrix.dot(diags(mu[:, i] / w) * B.T, p[:, i])

    Fnorm = np.linalg.norm(F, axis=1, ord=1)
    J = np.dot(w, Fnorm)

    return J, F, Fnorm


def get_gd_cost(Fnorm: np.ndarray, theta: float) -> float:
    """
    Compute Omega.

    :param Fnorm: 1-norm of fluxes
    :param theta: critical threshold for congestion
    :return: Omega
    """

    return 0.5 * np.sum(((Fnorm - theta) * np.heaviside(Fnorm - theta, 0)) ** 2)


def update_mu(
    mu: np.ndarray, p: np.ndarray, w: np.ndarray, B: csr_matrix, M: int, delta: float
) -> np.ndarray:
    """
    Update conductivities.

    :param mu: conductivities
    :param p: potential
    :param w: weights
    :param B: incidence matrix
    :param M: number of commodities
    :param delta: time step OT dynamics
    :return: conductivities
    """

    dp = B.T * p
    wtile = np.tile(w, (M, 1)).transpose()
    rhs = (mu * (dp**2)) / (wtile**2) - mu
    mu = mu + delta * rhs

    return mu


def compute_diff_cost(
    cost_new: float,
    cost: float,
    learning_rate: float,
) -> {float}:
    """
    Compute difference of cost between two consecutive time steps.

    :param cost_new: updated cost
    :param cost: outdated cost
    :param learning_rate: learning rate OT/GD
    :return: cost difference
    """

    return abs(cost - cost_new) / learning_rate
