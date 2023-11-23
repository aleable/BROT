# -*- coding: utf-8 -*-
"""
BROT (Bilevel Routing on networks with Optimal Transport) -- https://github.com/aleable/BROT

Contributors:
    Alessandro Lonardi
    Caterina De Bacco
"""

import os
import pickle as pkl
from argparse import ArgumentParser
from time import time

import networkx as nx
import numpy as np
from scipy.spatial import distance

from src.brot import BROT
from src.initialization import init, init_conductivities
from src.utils import assert_steps, print_main, assert_positivity_eps


def main():
    """
    BROT (Bilevel Routing on networks with Optimal Transport).

    ParsArgs:
        - ifolder: input data folder
        - ofolder: output data folder
        - V: verbose flag for BROT metadata
        - Vtime: verbose flag for execution time
        - tsev: time frequency for serialization of variables
        - topol: network topology
        - whichinflow: forcing type
        - stopol: seed for random synethic network topology
        - N: number of nodes
        - M: number of commodities
        - sinflows: seed for random inflows
        - sw: seed for noise Xi used to perturb weights
        - theta: critical threshold for congestion
        - delta: time step for OT dynamics
        - eta: learning rate for GD
        - lambda: convex combination weight for initialization of mu
        - TOT: time steps OT dynamics
        - TGD: time steps GD
        - totsteps: total time steps (> itstart for consistency with conv.)
        - itstart: number of steps after which convergence check starts
        - epsOT: threshold for convergence of J
        - epsGD: threshold for convergence of Omega
        - epsw: threshold for convergence of constraint
        - OTex: flag to execute OT dynamics
        - GDex: flag to execute GD
        - proj: projection method, needs to be one between "clipping", "momentum"
        - alpha: restituition coefficient for momentum
        - mask: mask for stochastic GD (q)
    """

    pars = ArgumentParser()

    pars.add_argument("-ifolder", "--input_folder", type=str, default="./data/input/")
    pars.add_argument(
        "-ofolder",
        "--output_folder",
        type=str,
        default="./data/output/",
    )
    pars.add_argument("-V", "--VERBOSE", type=lambda x: bool(int(x)), default=False)
    pars.add_argument("-Vtime", "--V_time", type=lambda x: bool(int(x)), default=False)
    pars.add_argument("-tsev", "--ts_exp_verb", type=int, default=1)
    pars.add_argument("-topol", "--whichtopol", type=str, default="disk")
    pars.add_argument("-whichinflow", "--whichinflow", type=str, default="dipoles_5")
    pars.add_argument("-stopol", "--seed_topol", type=int, default=0)
    pars.add_argument("-N", "--nodes_network", type=int, default=20)
    pars.add_argument("-M", "--commodities", type=int, default=5)
    pars.add_argument("-sinflows", "--seed_inflows", type=int, default=0)
    pars.add_argument("-sw", "--seed_weights", type=int, default=0)
    pars.add_argument("-theta", "--theta_capacity", type=float, default=0.1)
    pars.add_argument("-delta", "--delta_step", type=float, default=0.1)
    pars.add_argument("-eta", "--eta_step", type=float, default=0.1)
    pars.add_argument("-lambda", "--lambda_convex", type=float, default=0.05)
    pars.add_argument("-TOT", "--stop_time_OT", type=int, default=1)
    pars.add_argument("-TGD", "--stop_time_GD", type=int, default=1)
    pars.add_argument("-totsteps", "--tot_steps", type=int, default=2000)
    pars.add_argument("-itstart", "--it_check_start", type=int, default=20)
    pars.add_argument("-epsOT", "--eps_OT", type=float, default=1.0e-5)
    pars.add_argument("-epsGD", "--eps_GD", type=float, default=1.0e-5)
    pars.add_argument("-epsw", "--eps_w", type=float, default=1.0e-5)
    pars.add_argument("-OTex", "--exec_OT", type=lambda x: bool(int(x)), default=True)
    pars.add_argument("-GDex", "--exec_GD", type=lambda x: bool(int(x)), default=True)
    pars.add_argument("-proj", "--proj_method", type=str, default="clipping")
    pars.add_argument("-alpha", "--alpha_restituition", type=float, default=5)
    pars.add_argument("-mask", "--mask", type=float, default=1.0)

    args = pars.parse_args()

    topol = args.whichtopol
    whichinflow = args.whichinflow
    N = args.nodes_network
    M = args.commodities

    # dummy parameters -> extracted from pickled files
    if topol == "lattice" or topol == "disk" or topol == "euroroads":
        N = 0
        M = 0

    seed_network = args.seed_topol
    seed_g = args.seed_inflows
    seed_weights = args.seed_weights
    verbose = args.VERBOSE
    verbosetime = args.V_time
    OTexec = args.exec_OT
    GDexec = args.exec_GD
    delta = args.delta_step
    eta = args.eta_step
    theta_thresh = args.theta_capacity
    lambda_convex = args.lambda_convex
    TOT = args.stop_time_OT
    TGD = args.stop_time_GD
    epsOT = args.eps_OT
    epsGD = args.eps_GD
    epsw = args.eps_w
    proj_method = args.proj_method
    alpha_restituition = args.alpha_restituition
    mask = args.mask
    totsteps = args.tot_steps
    itcheckstart = args.it_check_start
    TIME_STEP_EXP_VERB = args.ts_exp_verb

    path = os.getcwd()
    ipath = path + "/" + args.input_folder
    opath = path + "/" + args.output_folder

    assert_steps(itcheckstart, totsteps)

    print("[ START ]")

    """
    INITIALIZATION
    """
    G, mu, w, S, g, commodities, M, sources_sinks = init(
        topol=topol,
        whichinflow=whichinflow,
        ipath=ipath,
        N=N,
        M=M,
        lambda_convex=lambda_convex,
        stopol=seed_network,
        sweight=seed_weights,
        ssynthinflows=seed_g,
        GDexec=GDexec,
        OTexec=OTexec,
    )

    print_main(verbosetime, 0, data={})

    # epsilon := 1% of minimimum weight
    epsilon = np.min(w) * 0.01
    assert_positivity_eps(epsilon)

    """
    BROT
    """
    brot = BROT(
        G,
        mu,
        w,
        S,
        M,
        epsilon=epsilon,
        delta=delta,
        eta=eta,
        theta=theta_thresh,
        TOT=TOT,
        TGD=TGD,
        totsteps=totsteps,
        epsOT=epsOT,
        epsGD=epsGD,
        epsw=epsw,
        OTexec=OTexec,
        GDexec=GDexec,
        proj_method=proj_method,
        alpha_restituition=alpha_restituition,
        mask=mask,
        verbose=verbose,
        it_check_start=itcheckstart,
        TIME_STEP_EXP_VERB=TIME_STEP_EXP_VERB,
    )

    """
    FIT
    """
    t1 = time()
    _ = brot.fit()
    t2 = time()

    """
    RE-RUN OT
    """
    _OT = dict()
    t1_OT, t2_OT = 0, 0
    if GDexec and not OTexec:

        """
        INITIALIZATION
        """
        # normalization of w
        w_convergence = _["wevol"] * normalization_lengths(G) / np.sum(_["wevol"])
        mu_convergence = init_conductivities(G, g, 1, S, M, sources_sinks)

        # epsilon := 1% of minimimum weight
        epsilon_OT = np.min(w_convergence) * 0.01
        assert_positivity_eps(epsilon)

        """
        BROT
        """
        brot_OT = BROT(
            G,
            mu_convergence,
            w_convergence,
            S,
            M,
            epsilon=epsilon_OT,
            delta=delta,
            eta=eta,
            theta=theta_thresh,
            TOT=TOT,
            TGD=TGD,
            totsteps=totsteps,
            epsOT=epsOT,
            epsGD=epsGD,
            epsw=epsw,
            OTexec=True,
            GDexec=False,
            proj_method=proj_method,
            alpha_restituition=alpha_restituition,
            mask=mask,
            verbose=verbose,
            it_check_start=itcheckstart,
            TIME_STEP_EXP_VERB=TIME_STEP_EXP_VERB,
        )

        """
        FIT
        """
        t1_OT = time()
        _OT = brot_OT.fit()
        t2_OT = time()

    elapsed_time = {"time": float(t2 - t1)}
    elapsed_time_OT = {"time": float(t2_OT - t1_OT)}
    print_main(verbosetime, 2, data=elapsed_time)
    print_main(verbosetime, 2, data=elapsed_time_OT)

    """
    SERIALIZATION
    """
    _["network"] = G
    _["forcing"] = S
    _["commodities"] = commodities
    metadata = dict()
    metadata["OTexec"] = OTexec
    metadata["GDexec"] = GDexec
    metadata["seed_topol"] = seed_network
    metadata["seed_weights"] = seed_weights
    metadata["N"] = N
    metadata["M"] = M
    metadata["seed_inflows"] = seed_g
    metadata["theta_thresh"] = theta_thresh
    metadata["delta"] = delta
    metadata["eta"] = eta
    metadata["TOT"] = TOT
    metadata["TGD"] = TGD
    metadata["epsOT"] = epsOT
    metadata["epsGD"] = epsGD
    metadata["lambda"] = lambda_convex

    results_fname = create_file_name(metadata, fname="results_")

    with open(
        opath + results_fname,
        "wb",
    ) as results_folder:
        pkl.dump(_, results_folder, protocol=pkl.HIGHEST_PROTOCOL)

    if GDexec and not OTexec:

        results_OT_fname = create_file_name(metadata, fname="results_OT_")

        with open(
            opath + results_OT_fname,
            "wb",
        ) as results_folder:
            pkl.dump(_OT, results_folder, protocol=pkl.HIGHEST_PROTOCOL)

    metadata_fname = create_file_name(metadata, fname="metadata_")

    with open(
        opath + metadata_fname,
        "wb",
    ) as results_folder:
        pkl.dump(metadata, results_folder, protocol=pkl.HIGHEST_PROTOCOL)

    print_main(verbosetime, 1, data=metadata)

    print("[ END ]")


def create_file_name(metadata: dict, fname: str) -> str:
    """
    Create file name for serialization of results and metadata.

    :param metadata: parameters used in BROT
    :param fname: string to append to save parameters and results separately
    :return: serialization file name
    """

    for md in list(metadata.items())[:]:
        fname += f"{md[0]}_{md[1]}_"
    fname = fname[: len(fname) - 1] + ".pkl"

    return fname


def normalization_lengths(G: nx.Graph):
    """
    Normalization constant for weights when running OT

    :param G: network
    :return: sum of weights
    """

    w = np.array(
        [
            distance.euclidean(G.nodes[edge[0]]["pos"], G.nodes[edge[1]]["pos"])
            for edge in G.edges()
        ]
    )

    return np.sum(w)


if __name__ == "__main__":
    main()
