# -*- coding: utf-8 -*-
"""
BROT (Bilevel Routing on networks with Optimal Transport) -- https://github.com/aleable/BROT

Contributors:
    Alessandro Lonardi
    Caterina De Bacco
"""

import pickle as pkl

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, distance

from .utils import assert_topol, assert_mass_balance


def init(
    topol: str,
    whichinflow: str,
    ipath: str,
    N: int,
    M: int,
    lambda_convex: float,
    stopol: int,
    sweight: int,
    ssynthinflows: int,
    GDexec: bool,
    OTexec: bool,
) -> {nx.Graph, np.ndarray, np.ndarray, np.ndarray}:
    """
    Initialization of G, w, S, g, commodities and mu.

    :param topol: network topology
    :param whichinflow: forcing type
    :param ipath: input data folder
    :param N: number of nodes
    :param M: number of commodities
    :param lambda_convex: convex combination weight for initialization of mu
    :param stopol: seed for random synethic network topology
    :param sweight: seed for noise Xi used to perturb weights
    :param ssynthinflows: seed for random inflows
    :param GDexec: flag to execute GD
    :param OTexec: flag to execute OT dynamics
    :return: graph, conductivities, weights, forcing, inflows, commodities, number sources, sources and sinks dict
    """

    assert_topol(topol)

    G = init_graph(ipath, topol, N, stopol)
    w = init_weights(G, sweight, OTexec, GDexec)
    S, g, commodities, M, sources_sinks = init_forcing(
        ipath, G, topol, M, whichinflow, ssynthinflows
    )
    mu = init_conductivities(G, g, lambda_convex, S, M, sources_sinks)

    return G, mu, w, S, g, commodities, M, sources_sinks


def init_graph(ipath: str, topol: str, N: int, stopol: int) -> {nx.Graph}:
    """
    Initialize network topology.

    :param ipath: input data folder
    :param topol: network topology
    :param N: number of nodes
    :param stopol: seed for random synethic network topology
    :return: None
    """

    # noinspection PyShadowingNames
    def place_nodes(G: nx.Graph, topol: str, N: int, stopol: int, ipath: str) -> None:
        """
        Add nodes and randomly place them, or extract their location.

        :param G: network
        :param topol: network topology
        :param N: number of nodes
        :param stopol: seed for random synethic network topology
        :param ipath: input data folder
        :return: None
        """

        if topol == "synthetic":
            np.random.seed(stopol)
            G.add_nodes_from(list(np.arange(N)))

            pos = np.random.uniform(low=0.0, high=1.0, size=(G.number_of_nodes(), 2))
            posdict = {i: (pos[i][0], pos[i][1]) for i in range(G.number_of_nodes())}
            nx.set_node_attributes(G, posdict, "pos")

        elif topol == "lattice" or topol == "disk":

            centers = pkl.load(open(ipath + f"coord_{topol}.pkl", "rb"))
            N = len(centers)
            G.add_nodes_from(list(np.arange(N)))
            pos = np.array([list(pp) for pp in centers])
            posdict = {i: (pos[i][0], pos[i][1]) for i in range(G.number_of_nodes())}
            nx.set_node_attributes(G, posdict, "pos")

        else:
            coord_f = open(ipath + f"coord_{topol}.dat", "r")
            lines = coord_f.readlines()
            G.add_nodes_from(list(np.arange(len(lines))))
            for i, line in enumerate(lines):
                G.nodes[i]["pos"] = tuple([float(xy) for xy in line.split()[0:2]])

        return None

    # noinspection PyShadowingNames
    def place_edges(G: nx.Graph, topol: str, ipath: str) -> None:
        """
        Add edges to network topology.

        :param G: network
        :param topol: network topology
        :param ipath: input data folder
        :return:
        """

        if topol == "synthetic":
            postri = [
                [G.nodes[i]["pos"][0], G.nodes[i]["pos"][1]]
                for i in range(G.number_of_nodes())
            ]
            tri = Delaunay(postri)
            for path in tri.simplices:
                nx.add_path(G, path)

        elif topol == "lattice" or topol == "disk":
            adjlist = pkl.load(open(ipath + f"adj_{topol}.pkl", "rb"))
            G.add_edges_from(adjlist)

        else:
            adj_f = open(ipath + f"adj_{topol}.dat", "r")
            lines = adj_f.readlines()
            adjlist = [np.array([line.split()[0:2]], dtype=int)[0] for line in lines]
            G.add_edges_from(adjlist)

        return None

    G = nx.Graph()
    place_nodes(G, topol, N, stopol, ipath)
    place_edges(G, topol, ipath)

    return G


def init_weights(G: nx.Graph, sweight: int, OTexec: bool, GDexec: bool) -> np.ndarray:
    """
    Initialize the weights of the edges as Euclidian distance of nodes, added to noise Xi.

    :param G: network
    :param sweight: seed for noise Xi used to perturb weights
    :param OTexec:  flag to execute OT dynamics
    :param GDexec: flag to execute GD
    :return: the initialized weights
    """

    np.random.seed(sweight)

    w = np.array(
        [
            distance.euclidean(G.nodes[edge[0]]["pos"], G.nodes[edge[1]]["pos"])
            for edge in G.edges()
        ]
    )

    for we, e in list(zip(w, G.edges())):
        G[e[0]][e[1]]["w_denoised"] = we

    minw = np.min(w)
    xi = np.random.rand(G.number_of_edges())
    xi -= np.mean(xi)
    xi /= max(abs(xi))
    xi *= minw * 0.1

    if (OTexec is True) and (GDexec is not True):
        pass
    else:
        w = w + xi

    return w


def init_conductivities(
    G: nx.Graph,
    g: np.ndarray,
    lambda_convex: float,
    S: np.ndarray,
    M: int,
    sources_sinks: dict,
) -> np.ndarray:
    """
    Initialize the conductivities.

    :param G: network
    :param g: inflows
    :param lambda_convex: convex combination weight for initialization of mu
    :param S: forcing
    :param M: number of  commodities
    :param sources_sinks: sources and sinks dict
    :return: conductivities
    """

    # noinspection PyShadowingNames
    def dijskstra_init(
        G: nx.Graph,
        S: np.ndarray,
        M: int,
        sources_sinks: dict,
    ) -> np.ndarray:
        """
        Use power law mu ~ |F| combined with Dijkstra to initialize muSP.

        :param G: network
        :param S: forcing
        :param M: number of commodities
        :param sources_sinks: sources and sinks dict
        :return: conductivities for shortest path
        """

        paths = dict()
        fluxes_paths = np.zeros((G.number_of_edges(), M))
        sources = np.array(list(sources_sinks.keys()))

        for source in sources:
            for target in sources_sinks[source]:
                if source != target:

                    path = nx.dijkstra_path(G, source, target, weight="w_denoised")
                    paths[tuple((source, target))] = [
                        (e[0], e[1]) for e in list(zip(path, path[1:]))
                    ]

                    for e in paths[tuple((source, target))]:
                        try:
                            idx = list(G.edges()).index(tuple((e[0], e[1])))
                            index_source = np.where(sources == source)[0][0]
                            fluxes_paths[idx, index_source] += abs(
                                S[target, index_source]
                            )
                        except ValueError:
                            idx = list(G.edges()).index(tuple((e[1], e[0])))
                            index_source = np.where(sources == source)[0][0]
                            fluxes_paths[idx, index_source] += abs(
                                S[target, index_source]
                            )

        musp = abs(fluxes_paths)

        return musp

    muFF = np.ones((G.number_of_edges(), M)) * g
    muSP = dijskstra_init(G, S, M, sources_sinks)

    mu_init = lambda_convex * muFF + (1 - lambda_convex) * muSP

    return mu_init


def init_forcing(
    ipath: str, G: nx.Graph, topol: str, M: int, whichinflow: str, ssynthinflows: int
) -> {np.ndarray, np.ndarray, np.ndarray}:
    """
    Initialization of forcing.

    Sources for:
        - synthetic networks are M random nodes
        - manual network are M nodes chosen manually
        - real networks are the M nodes with largest g

    :param ipath: input data folder
    :param G: network
    :param topol: network topology
    :param M: number of  commodities
    :param whichinflow: forcing type
    :param ssynthinflows: seed for random inflows
    :return: forcing, inflows, commodities, number of sources, sources and sinks dict
    """

    # noinspection PyShadowingNames
    def generate_synth_inflows(M: int, ssynthinflows: int) -> np.ndarray:
        """
        Generate inflows for synthetic network.

        :param M: number of commodities
        :param ssynthinflows: seed for random inflows
        :return: inflows
        """

        np.random.seed(ssynthinflows)
        g = np.random.uniform(low=0, high=1, size=M)
        g /= np.sum(g)

        return g

    # noinspection PyShadowingNames
    def influence_assignment(
        commodities: np.ndarray, M: int, g: np.ndarray
    ) -> np.ndarray:
        """
        Initialize forcing from inflows using influence assignment of:
        https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.043010

        :param commodities: commodities
        :param M: number of commodities
        :param g: inflows
        :return: forcing
        """

        S = np.zeros((G.number_of_nodes(), M))

        for i in range(M):
            gsum = np.sum(g)
            gsum -= g[i]
            r = g / gsum
            S[commodities, i] = -r * g[i]
            S[commodities[i], i] = g[i]

        return S

    if topol == "synthetic":
        commodities = np.array(G.nodes)[:M]
        sources_sinks = {
            commodities[i]: list(commodities[:i]) + list(commodities[i + 1:])
            for i in range(M)
        }
        g = generate_synth_inflows(M, ssynthinflows)
        S = influence_assignment(commodities, M, g)

    elif topol == "lattice" or topol == "disk":

        sources_sinks = pkl.load(open(ipath + f"sources_sinks_{whichinflow}.pkl", "rb"))

        commodities = list()
        for item in sources_sinks.items():
            commodities.append(item[0])
            for s in item[1]:
                commodities.append(s)

        M = len(sources_sinks.keys())
        S = pkl.load(open(ipath + f"S_{whichinflow}.pkl", "rb"))
        g = S[S > 0]

    else:
        sources_sinks = pkl.load(open(ipath + f"sources_sinks_{whichinflow}.pkl", "rb"))

        commodities = list()
        g = list()
        g_f = open(ipath + f"g_{topol}.dat", "r")
        lines = g_f.readlines()
        for i, line in enumerate(lines):
            commodities.append([float(xy) for xy in line.split()[0:2]][0])
            g.append([float(xy) for xy in line.split()[0:2]][1])
        commodities = np.array(commodities, dtype=int)
        g = np.array(g, dtype=float)

        M = len(g)
        S = influence_assignment(commodities, M, g)

    assert_mass_balance(S)

    return S, g, commodities, M, sources_sinks
