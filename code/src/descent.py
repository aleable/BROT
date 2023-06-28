# -*- coding: utf-8 -*-
"""
BROT (Bilevel Routing on networks with Optimal Transport) -- https://github.com/aleable/BROT

Contributors:
    Alessandro Lonardi
    Caterina De Bacco
"""

import sys

import numpy as np
from scipy.sparse import csr_matrix, diags

from .dynamics import compute_diff_cost, get_potential, get_gd_cost
from .utils import (
    assert_mass_conservation,
    assert_positivity_w,
    print_descent,
    assert_projection,
)


def gd(
    mu: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    B: csr_matrix,
    N: int,
    E: int,
    M: int,
    theta: float,
    eta: float,
    TGD: int,
    proj_method: str,
    epsilon: float,
    alpha: float,
    relax: float,
    psi: np.ndarray,
    psievol: list,
    verbose: bool,
    OTexec: bool,
    step: int,
    TSEV: int,
) -> {np.ndarray, np.ndarray, list, float, list, list, list, list}:
    """
    GD.

    :param mu: conductivities
    :param w: weights
    :param S: forcing
    :param B: incidence matrix
    :param N: number of nodes
    :param E: number of edges
    :param M: number of commodities
    :param theta: critical threshold for congestion
    :param eta: learning rate GD
    :param TGD: time steps GD
    :param proj_method: rojection method
    :param epsilon: constraint of weights
    :param alpha: restituition coefficient
    :param relax: relaxation for Laplacian inversion
    :param psi: gradient
    :param psievol: gradient evolution
    :param verbose: verbose flag for BROT metadata
    :param OTexec: flag to execute OT dynamics
    :param step: time step
    :param TSEV: time frequency for serialization of variables
    :return: weights, gradient, gradient evol, Omega difference, {J, Omega, mu, w, F} evolution
    """

    print_descent(verbose, step, TSEV)

    # INITIALIZATION
    assert_projection(proj_method)

    it = 0
    p = get_potential(mu, w, S, B, N, M, relax)
    F, Fnorm = get_flux(mu, w, p, B, E, M)
    Omega = get_gd_cost(Fnorm, theta)

    dOmega = sys.float_info.max
    Jstack = list()
    Omegastack = list()
    wstack = list()
    mustack = list()
    Fstack = list()

    if (not OTexec) and (step == 0):
        J = get_ot_cost(Fnorm, w)
        Omegastack.append(Omega)
        Jstack.append(J)
        wstack.append(w)
        mustack.append(mu)
        Fstack.append(F)
        it = 1

    # UPDATE
    while it < TGD:

        w, psi, psievol = update_w(
            w,
            F,
            Fnorm,
            mu,
            eta,
            theta,
            B,
            E,
            M,
            psievol,
            proj_method,
            epsilon,
            alpha,
            step,
            TSEV,
        )

        p = get_potential(mu, w, S, B, N, M, relax)
        F, Fnorm = get_flux(mu, w, p, B, E, M)
        Omega_new = get_gd_cost(Fnorm, theta)
        dOmega = compute_diff_cost(Omega_new, Omega, eta)

        Omega = Omega_new

        Omegastack.append(Omega)
        wstack.append(w)
        mustack.append(mu)
        Fstack.append(F)
        J = get_ot_cost(Fnorm, w)
        Jstack.append(J)

        assert_mass_conservation(B, F, S)

        it += 1

    return w, psi, psievol, dOmega, Jstack, Omegastack, mustack, wstack, Fstack


def update_w(
    w: np.ndarray,
    F: np.ndarray,
    Fnorm: np.ndarray,
    mu: np.ndarray,
    eta: float,
    theta: float,
    B: csr_matrix,
    E: int,
    M: int,
    psievol: list,
    proj_method: str,
    epsilon: float,
    alpha: float,
    step: int,
    TSEV: int,
) -> {np.ndarray, np.ndarray}:
    """
    Update weights.

    Sizes of variables:
        - |delta_tile| = ExExM
        - |Ls| = NxNxM
        - |Lps| = NxNxM
        - |GLap| = ExExM

    :param w: weights
    :param F: fluxes
    :param Fnorm: 1-norm of fluxes
    :param mu: conductivities
    :param eta: learning rate
    :param theta: critical threshold for congestion
    :param B: incidence matrix
    :param E: number of edges
    :param M: number of commodities
    :param psievol: gradient evolution
    :param proj_method: projection method
    :param epsilon: constraint of weights
    :param alpha: restituition coefficient
    :param step: time step
    :param TSEV: ime frequency for serialization of variables
    :return: weights, gradient, gradient evolution
    """

    # noinspection PyShadowingNames
    def get_gradients(
        w: np.ndarray,
        F: np.ndarray,
        Fnorm: np.ndarray,
        mu: np.ndarray,
        theta: float,
        B: csr_matrix,
        E: int,
        M: int,
    ) -> np.ndarray:
        """
        Compute gradient of Omega.

        :param w: weights
        :param F: fluxes
        :param Fnorm: 1-norm of fluxes
        :param mu: conductivities
        :param theta: critical threshold for congestion
        :param B: incidence matrix
        :param E: number of edges
        :param M: number of commodities
        :return: gradient
        """

        delta_tile = np.zeros((E, E, M))
        GLap = np.zeros((E, E, M))

        Delta = (Fnorm - theta) * np.heaviside(Fnorm - theta, 0)
        deltaedge = np.identity(E)

        for i in range(M):
            delta_tile[:, :, i] = deltaedge
            Li = B * diags(mu[:, i] / w) * B.T
            Lpsi = np.linalg.pinv(Li.todense(), rcond=1e-15, hermitian=True)
            GLap[:, :, i] = B.T * Lpsi * B

        chainrule1 = np.einsum("li,l,li,lei->lei", mu, 1 / w, np.sign(F), GLap)
        chainrule2 = np.einsum("ei,lei->lei", np.sign(F), delta_tile)

        psi = np.einsum("l,ei,e,lei->e", Delta, F, 1 / w, chainrule1 - chainrule2)

        return psi

    psi = get_gradients(w, F, Fnorm, mu, theta, B, E, M)

    if proj_method == "momentum":
        psi = project_with_momentum(w, psi, alpha, epsilon)

    w = w - eta * psi

    if proj_method == "clipping":
        w = project_with_clipping(w, epsilon)

    assert_positivity_w(w)

    if step % TSEV == 0:
        psievol.append(psi)

    return w, psi, psievol


def project_with_clipping(w: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Project weights by clipping them as w >= epsilon.

    :param w: weights
    :param epsilon: constraint of weights
    :return: clipped weights
    """
    return np.clip(w, epsilon, None, out=w)


def project_with_momentum(
    w: np.ndarray, psi: np.ndarray, alpha: float, epsilon: float
) -> np.ndarray:
    """
    Project weights by adding momentum, inspired by:
    https://www.jmlr.org/papers/v23/21-0798.html

    :param w: weights
    :param psi: gradient
    :param alpha: restituition coefficient
    :param epsilon: constraint of weights
    :return: modified gradients with momentum
    """

    cond1 = -psi < -alpha * (w - epsilon)
    cond2 = w < epsilon
    idxproj = np.where(np.logical_and(cond1, cond2))[0]
    fhat = psi
    fhat[idxproj] = -alpha * (epsilon - w[idxproj])

    return fhat


def get_flux(
    mu: np.ndarray, w: np.ndarray, p: np.ndarray, B: csr_matrix, E: int, M: int
) -> {np.ndarray, np.ndarray}:
    """
    Compute fluxes.

    :param mu: conductivities
    :param w: weights
    :param p: potential
    :param B: incidence matrix
    :param E: number of edges
    :param M: number of commodities
    :return: fluxes, 1-norm of fluxes
    """

    F = np.zeros((E, M))

    for i in range(M):
        F[:, i] = csr_matrix.dot(diags(mu[:, i] / w) * B.T, p[:, i])

    Fnorm = np.linalg.norm(F, axis=1, ord=1)

    return F, Fnorm


def get_ot_cost(Fnorm: np.ndarray, w: np.ndarray) -> {float}:
    """
    Compute J.

    :param Fnorm: 1-norm of fluxes
    :param w: weights
    :return: J
    """

    return np.dot(w, Fnorm)
