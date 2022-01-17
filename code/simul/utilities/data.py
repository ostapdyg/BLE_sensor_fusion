import dataclasses
import json
import os

import numpy as np
from git.repo import Repo
from simul.parameters import Parameters


def dump_experiment(
    exp_name: str,
    params: Parameters,
    dist: np.ndarray,
    signals_data: np.ndarray,
    dist_probs: np.ndarray = None,
):
    assert os.path.exists("../data")
    path = os.path.join("../data", exp_name)
    if os.path.exists(path):
        if input(f"{path} exists, rewrite? (y/n)") != "y":
            return
    else:
        os.mkdir(path)
    with open(os.path.join(path, "hash.txt"), "w") as fp:
        repo = Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        fp.write(sha)
    with open(os.path.join(path, "params.json"), "w") as fp:
        json.dump(dataclasses.asdict(params), fp)
    with open(os.path.join(path, "dist.npy"), "wb") as fp:
        np.save(fp, dist)
    with open(os.path.join(path, "signals_data.npy"), "wb") as fp:
        np.save(fp, signals_data)
    with open(os.path.join(path, "dist_probs.npy"), "wb") as fp:
        np.save(fp, dist_probs)


def load_experiment(exp_name: str):
    assert os.path.exists("../data")
    path = os.path.join("../data", exp_name)
    assert os.path.isdir(path)
    with open(os.path.join(path, "params.json"), "r") as fp:
        params = Parameters(**json.load(fp))
    with open(os.path.join(path, "dist.npy"), "rb") as fp:
        dist = np.load(fp)
    with open(os.path.join(path, "signals_data.npy"), "rb") as fp:
        signals_data = np.load(fp)
    with open(os.path.join(path, "dist_probs.npy"), "rb") as fp:
        dist_probs = np.load(fp)

    return params, dist, signals_data, dist_probs
