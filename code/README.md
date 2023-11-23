![alt text](https://github.com/aleable/BROT/blob/main/misc/logo.svg)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Implementation details

### Table of Contents  

- [Implementation details](#implementation-details)
  - [Table of Contents](#table-of-contents)
- [What's included](#whats-included)
- [Requirements](#requirements)
- [How to use](#how-to-use)
  - [Parameters](#parameters)
- [I/O format](#io-format)
  - [Input](#input)
  - [Output](#output)
- [Usage examples](#usage-examples)
- [Contacts](#contacts)


## What's included

- ```dashboard.ipynb```: Jupyter notebook containing an easy-to-use interface with BROT
- ```dashboard_utils.ipynb```: utils needed by ```dashboard.ipynb```
- ```main.py```: main function of BROT
- ```scr/```
  - ```brot.py```: BROT class
  - ```descent.py```: PSGD implementation
  - ```dynamics.py```: Optimal Transport ODEs integration
  - ```initialization.py```: problem setup
  - ```utils.py```: utilities for code metadata


## Requirements

All the dependencies needed to run the algorithms can be installed using ```setup.py```.
This script can be executed with the command:

```bash
python setup.py
```

Now, you are ready to use the code! To do so, you can simply use the notebook ```dashboard.ipynb```, from which you can access BROT and run it (in its default configuration) with the magic command ```%run main.py```.<br/>

## How to use

### Parameters

The parameters you can pass to BROT are:

```-ifolder``` (type=str, default="./data/input/"): input data folder<br>
```-ofolder``` (type=str, default="./data/output/"): output data folder<br>
```-V``` (type=lambda x: bool(int(x)), default=False): verbose flag for BROT metadata<br>
```-Vtime``` (type=lambda x: bool(int(x)), default=False): verbose flag for elapsed time<br>
```-tsev``` (type=int, default=1): time frequency for serialization of variables (to set manually in ```brot.py```)<br>
```-topol``` (type=str, default="disk"): network topology<br>
```-whichinflow``` (type=str, default="center_to_rim_8"): inflow<br>
```-stopol``` (type=int, default=0): seed for random synthetic topology<br>
```-N``` (type=int, default=20): number of nodes<br>
```-M``` (type=int, default=5): number of OD pairs<br>
```-sinflows``` (type=int, default=0): see for random inflow<br>
```-sw``` (type=int, default=0): seed for noise added to weights in PSGD step<br>
```-theta``` (type=float, default=0.1): congestion threshold<br>
```-delta``` (type=float, default=0.1): time step for OT integration<br>
```-eta``` (type=float, default=0.1): learning rate for PSGD<br>
```-lambda``` (type=float, default=1): convex combination weight for μ<br>
```-TOT``` (type=int, default=1): consecutive time steps OT dynamics<br>
```-TGD``` (type=int, default=1): consecutive time steps PSGD<br>
```-totsteps``` (type=int, default=2000): total number of time steps<br>
```-itstart``` (type=int, default=20): number of steps after which convergence check starts<br>
```-epsOT``` (type=float, default=1.0e-5): threshold for convergence of J<br>
```-epsGD``` (type=float, default=1.0e-5): threshold for convergence of Ω<br>
```-epsw``` (type=float, default=1.0e-5): threshold for convergence of w<br>
```-OTex``` (type=lambda x: bool(int(x)), default=True): flag to run OT dynamics<br>
```-GDex``` (type=lambda x: bool(int(x)), default=True): flag to run PSGD<br>
```-proj``` (type=str, default="clipping"): projection method (```"clipping"```, or ```"momentum"``` to use momentum-based projection [1])<br>
```-alpha``` (type=float, default=5): restituition coefficient [1]<br>
```-mask``` (type=float, default=1.0): mask for stochastic GD, i.e., probaility q for gradients dropout

[1] Michael Muehlebach, Michael I. Jordans, <a href="https://www.jmlr.org/papers/volume23/21-0798/21-0798.pdf"> Journal of Machine Learning Research 23, 1 (2022)</a>.

## I/O format

### Input

BROT can be tested on four different networks, which can be chosen with the ```-topol``` parameter. These are:
- ```-topol = "synthetic"```: random points in the unitary square joined by their Delaunay triangulation (|V| = ```-N```, |E| = function of ```-N```)
- ```-topol = "lattice"```: hexagonally shaped triangular lattice (|V| = 61, |E| = 156)
- ```-topol = "disk"```: random points in the unitary disk joined by their Delaunay triangulation (|V| = 300, |E| = 864)
- ```-topol = "euroroads"```: Post-processed Euroroads network [2] (|V| = 541, |E| = 712)

Inflows can be changed with ```-whichinflow```. In order to select inflows that are compatible with the topologies listed above, one should set:

- ```-topol = "synthetic"```: ```synthetic```<br>This assign a random inflow to ```-M``` random nodes, and computes their outflows using an influence assignment scheme [3]
- ```-topol = "lattice"```: ```lattice```<br>This sets a positive inflow (+1/2) to one node only, and its correspondent outflow (-1/2) to another node
- ```-topol = "disk"```: ```center_to_rim_```{4,8,16}, ```dipoles_```{5,10,15}<br>These set center-to-rim OD pairs structures as described in the manuscript, with the possibility of having D = 16. Additionally, one can experiment with ```dipoles_```{5,10,15}, to test a setup where 5, 10, 15 ODs are paired together as distinguishable start-end points
- ```-topol = "euroroads"```: ```euroroads```<br>Inflows are 15 densely populated cities in Europe, outflows are computed using an influence assignment scheme [3]

[2] Jérôme Kunegis, <a href="https://dl.acm.org/doi/abs/10.1145/2487788.2488173"> Proceedings of the 22nd International Conference on World Wide Web (2013)</a>.<br>
[3] Alessandro Lonardi, Enrico Facca, Mario Putti, and Caterina De Bacco, <a href="https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.043010">Phys. Rev. Research 3, 043010 (2021)</a>.

### Output

All outputs of ```brot.py``` are serialized in ```main.py```. The dictionary of results is made of the following items:

```python
results["Jevol"] = np.array(sum(Jevol, []))[-1] # J at convergence
results["Omegaevol"] = np.array(sum(Omegaevol, []))[-1] # Ω at convergence
results["muevol"] = np.array(sum(muevol, []))[-1] # μ at convergence
results["wevol"] = np.array(sum(wevol, []))[-1] # w at convergence
results["Fevol"] = np.array(sum(Fevol, []))[-1] # F at convergence
results["network"] = G # network
results["forcing"] = S # forcing of OT dynamics
results_["commodities"] = commodities # nodes used in OD pairs
```

In order serialize heavier outputs, with the time-evolution of the main variables, manually modify ```brot.py```.

## Usage examples

A usage example of BROT can be found in ```dashboard.ipynb```.

## Contacts

For any issues or questions, feel free to contact us sending an email to <a href="alessandro.lonardi@tuebingen.mpg.de">alessandro.lonardi@tuebingen.mpg.de</a>.
