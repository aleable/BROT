![alt text](https://github.com/aleable/BROT/blob/main/misc/logo.svg)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

___

BROT 🍞 (**B**ilevel **R**outing on networks with **O**ptimal **T**ransport) is a Python implementation of the algorithms used in:

- [1] Alessandro Lonardi and Caterina De Bacco. <i>Bilevel Optimization for Traffic Mitigation in Optimal Transport Networks</i>. Physical Review Letters [<a href="https://arxiv.org/abs/2306.16246">arXiv</a>].

This is a scheme capable of extracting origin-destination paths on networks by making a trade off between transportation efficiency and over-trafficked links. The core algorithm alternates the integration of a system of ODEs to find passengers' shortest origin-destination routes, and Projected Stochastic Gradient Descent to mitigate traffic.

**If you use this code please cite [1].**

## What's included

- ```code```: contains the all the scripts necessary to run BROT, and a Jupyter notebook (```dashboard.ipynb```) with a tutorial on how to use our code
- ```data/input```: contains the data needed to test the scheme on both synthetic topologies and on the [Euroroads network](http://konect.cc/networks/subelj_euroroad/) [2]
- ```data/output```: folder for data serialization
- ```misc```: files used for the README.md
- ```setup.py```: setup file to build the Python environment

[2] Jérôme Kunegis, <a href="https://dl.acm.org/doi/abs/10.1145/2487788.2488173"> Proceedings of the 22nd International Conference on World Wide Web (2013)</a>.<br/>

## How to use

To download this repository, copy and paste the following:

```bash
git clone https://github.com/aleable/BROT
```

**You are ready to test the code! But if you want to know how click [HERE](https://github.com/aleable/BROT/tree/main/code)**.

## Contacts

For any issues or questions, feel free to contact us sending an email to <a href="alessandro.lonardi@tuebingen.mpg.de">alessandro.lonardi@tuebingen.mpg.de</a>.
