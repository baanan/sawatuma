# Sawatuma

A collaborative filtering music recommender built on the [LFM-2b](http://www.cp.jku.at/datasets/LFM-2b/) dataset.

The actual implementation uses a type of [Matrix Factorization](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) that accepts implicit data (such as the amount of times a user has listened to a track) as described by [Yifan Hu et al.](http://yifanhu.net/PUB/cf.pdf) This was chosen as the dataset lists listening events, not actual ratings of tracks.

The main bits of code are described in four files:
- [`src/sawatuma/model.py`]: the matrix factorization model
- [`src/sawatuma/datasets.py`]: the code responsible for downloading, extracting, and portioning the data
- [`src/__main__.py`]: describes the main parameters of the model
- [`src/tui.py`]: the basic tui that comes with the model

## Installation

This project requires pdm, so please [install it first](https://pdm-project.org/en/latest/#installation).

First, clone the reposistory:

```bash
git clone https://github.com/baanan/sawatuma.git
cd sawatuma
```

If desired, you may want to download a precompiled model and track mapping from the [latest release](https://github.com/baanan/sawatuma/releases/latest). If you do, place both of these files in a folder named `data` off of the project root.

Then, run the `__main__` file using pdm (this may take a very long time!):

```bash
pdm run src/__main__.py
```

