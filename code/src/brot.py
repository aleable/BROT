# -*- coding: utf-8 -*-
"""
BROT (Bilevel Routing on networks with Optimal Transport) -- https://github.com/aleable/BROT

Contributors:
    Alessandro Lonardi
    Caterina De Bacco
"""

import sys

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from .descent import gd
from .dynamics import otdyn
from .utils import print_brot


class BROT:
    """
    BROT architecture.
    BROT is composed of a single function fit() which executes the OT dynamics and/or the descent scheme.
    """

    def __init__(
        self,
        G: nx.Graph,
        mu: np.ndarray,
        w: np.ndarray,
        S: np.ndarray,
        M: int,
        epsilon: float,
        delta: float,
        eta: float,
        theta: float,
        TOT: int,
        TGD: int,
        totsteps: int,
        epsOT: float,
        epsGD: float,
        epsw: float,
        OTexec: bool,
        GDexec: bool,
        proj_method: str,
        alpha_restituition: float,
        verbose: bool,
        it_check_start: int,
        TIME_STEP_EXP_VERB: int,
    ):
        """
        :param G: network
        :param mu: conductivities
        :param w: weights
        :param S: forcing
        :param M: number of commodities
        :param epsilon: threshold for projection of w
        :param delta: time step for OT dynamics
        :param eta: learning rate for GD
        :param theta: critical threshold for congestion
        :param TOT: time steps OT dynamics in BROT
        :param TGD: time steps GD in BROT
        :param totsteps: total time steps
        :param epsOT: threshold for convergence of J
        :param epsGD: threshold for convergence of Omega
        :param epsw: threshold for convergence of constraint
        :param OTexec: flag to execute OT dynamics
        :param GDexec: flag to execute GD
        :param proj_method: projection method
        :param alpha_restituition: alpha
        :param verbose: verbose flag for BROT metadata
        :param it_check_start: starting step for convergence check
        :param TIME_STEP_EXP_VERB: time frequency for serialization
        """

        self.G = G
        self.mu = mu
        self.w = w
        self.S = S
        self.M = M
        self.epsilon = epsilon
        self.delta = delta
        self.eta = eta
        self.theta = theta
        self.TOT = TOT
        self.TGD = TGD
        self.totsteps = totsteps
        self.epsOT = epsOT
        self.epsGD = epsGD
        self.epsw = epsw
        self.OTexec = OTexec
        self.GDexec = GDexec
        self.proj_method = proj_method
        self.alpha_restituition = alpha_restituition
        self.verbose = verbose
        self.it_check_start = it_check_start
        self.TSEV = TIME_STEP_EXP_VERB

    def fit(self) -> dict:
        """
        Running BROT.

        :return: BROT variables at convergence
        """

        N = self.G.number_of_nodes()
        E = self.G.number_of_edges()
        B = csr_matrix(
            nx.incidence_matrix(
                self.G, nodelist=list(range(self.G.number_of_nodes())), oriented=True
            )
        )
        M = self.M

        relax = 1e-12
        step = 0
        dJ = sys.float_info.max
        dOmega = sys.float_info.max
        psi = np.zeros(E)
        Jevol = list()
        Omegaevol = list()
        muevol = list()
        wevol = list()
        Fevol = list()
        psievol = list()
        results = dict()
        conv = False

        while not conv:

            # set OT/GD convergence to True if they are not executed
            if not self.OTexec:
                dJ = 0
            if not self.GDexec:
                dOmega = 0

            if self.OTexec:

                print_brot(
                    self.verbose, 0, step, c=0, dc=0, TIME_STEP_EXP_VERB=self.TSEV
                )

                # OT dynamics
                self.mu, dJ, Jstack, Omegastack, mustack, wstack, Fstack = otdyn(
                    self.mu,
                    self.w,
                    self.S,
                    B,
                    N,
                    E,
                    M,
                    self.theta,
                    self.delta,
                    self.TOT,
                    relax,
                    self.verbose,
                    step,
                    self.GDexec,
                    self.TSEV,
                )

                # lightweight serialization
                if step % self.TSEV == 0:
                    Jevol.append(Jstack)
                    Omegaevol.append(Omegastack)
                    Fevol.append(Fstack)
                    muevol.append(mustack)
                    wevol.append(wstack)

                print_brot(
                    self.verbose,
                    1,
                    step,
                    Jstack[-1],
                    dJ,
                    TIME_STEP_EXP_VERB=self.TSEV,
                )

                conv = check_convergence(
                    step,
                    dJ,
                    dOmega,
                    self.epsilon,
                    self.w,
                    self.it_check_start,
                    self.epsOT,
                    self.epsGD,
                    self.epsw,
                    self.totsteps,
                )

                step += 1

            if self.GDexec:

                print_brot(
                    self.verbose,
                    2,
                    step,
                    c=0,
                    dc=0,
                    TIME_STEP_EXP_VERB=self.TSEV,
                )

                # GD
                (
                    self.w,
                    psi,
                    psievol,
                    dOmega,
                    Jstack,
                    Omegastack,
                    mustack,
                    wstack,
                    Fstack,
                ) = gd(
                    self.mu,
                    self.w,
                    self.S,
                    B,
                    N,
                    E,
                    M,
                    self.theta,
                    self.eta,
                    self.TGD,
                    self.proj_method,
                    self.epsilon,
                    self.alpha_restituition,
                    relax,
                    psi,
                    psievol,
                    self.verbose,
                    self.OTexec,
                    step,
                    self.TSEV,
                )

                # lightweight serialization
                if step % self.TSEV == 0:
                    Jevol.append(Jstack)
                    Omegaevol.append(Omegastack)
                    Fevol.append(Fstack)
                    muevol.append(mustack)
                    wevol.append(wstack)

                print_brot(self.verbose, 1, step, Omegastack[-1], dOmega, self.TSEV)

                conv = check_convergence(
                    step,
                    dJ,
                    dOmega,
                    self.epsilon,
                    self.w,
                    self.it_check_start,
                    self.epsOT,
                    self.epsGD,
                    self.epsw,
                    self.totsteps,
                )

                step += 1

        # heavy serialization
        """
        results["Jevol"] = np.array(sum(Jevol, []))
        results["Omegaevol"] = np.array(sum(Omegaevol, []))
        results["muevol"] = np.array(sum(muevol, []))
        results["wevol"] = np.array(sum(wevol, []))
        results["Fevol"] = np.array(sum(Fevol, []))
        results["psievol"] = np.array(psievol)
        """

        # lightweight serialization, only variable at convergence
        results["Jevol"] = np.array(sum(Jevol, []))[-1]
        results["Omegaevol"] = np.array(sum(Omegaevol, []))[-1]
        results["muevol"] = np.array(sum(muevol, []))[-1]
        results["wevol"] = np.array(sum(wevol, []))[-1]
        results["Fevol"] = np.array(sum(Fevol, []))[-1]
        try:
            results["psievol"] = np.array(psievol)[-1]
        except IndexError:
            results["psievol"] = 0

        return results


def check_convergence(
    step: int,
    dJ: float,
    dOmega: float,
    epsilon: float,
    weights: np.ndarray,
    it_check_start: int,
    epsOT: float,
    epsGD: float,
    epsw: float,
    totsteps: int,
) -> bool:
    """
    Check convergence of BROT.

    :param step: time step for OT dynamics / learning rate for GD
    :param dJ: difference of J between two consecutive time steps
    :param dOmega: difference of Omega between two consecutive time steps
    :param epsilon: threshold for projection of w
    :param weights: weights
    :param it_check_start: starting step for convergence check
    :param epsOT: threshold for convergence of J
    :param epsGD: threshold for convergence of Omega
    :param epsw: threshold for convergence of constraint
    :param totsteps: total time steps
    :return: convergence state flag
    """

    conv = False
    conv_1 = False
    conv_2 = False

    if step > totsteps:
        conv = True

    if step > it_check_start:
        if (dJ < epsOT) and (dOmega < epsGD):
            conv_1 = True

    if np.min(weights) >= epsilon:
        conv_2 = True
    else:
        if epsilon - np.min(weights[weights < epsilon]) < epsw:
            conv_2 = True

    if conv_1 and conv_2:
        conv = True

    return conv
