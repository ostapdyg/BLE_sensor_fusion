import numpy as np

from simul.parameters import Parameters


def signal_add_noise(signals:np.ndarray, val:float = 1, kind:str = "normal") -> np.ndarray:
    if(kind == "normal"):
        return signals + np.random.normal(0, val, signals.shape) + 1j*np.random.normal(0, val, signals.shape)
    if(kind == "uniform"):
        return signals + np.random.uniform(-val, val, signals.shape) + 1j*np.random.uniform(-val, val, signals.shape)


def signal_prune_default(signals:np.ndarray) -> np.ndarray:
    # for ts_i in range(signals):
    #     f2_i = np.floor(ts_i / 8) % 5
    #     f_idx = (ts_i % 8) * 10 + f2_i * 2
    ts_idxs = np.arange(signals.shape[1], dtype=np.int32)
    f2_idxs = np.floor(ts_idxs / 8) % 5
    f_idxs = np.int32((ts_idxs %8) * 5 + f2_idxs)

    res = np.empty(signals.shape, dtype=np.complex128)
    res[:] = np.nan
    for t_idx in np.arange(signals.shape[1]):
        res[f_idxs[t_idx], t_idx] = signals[f_idxs[t_idx], t_idx]
    return res

def signal_prune(signals:np.ndarray, kind:str = "default") -> np.ndarray:
    if(kind == "default"):
        return signal_prune_default(signals)
    pass


if(__name__ == "__main__"):
    from distance_determination import estimate_dist, simulate_signals
    from run_experiment import experiments

    exp_name = "default_full"
    params = experiments[exp_name]
    dist, signal_full = simulate_signals(params)
    signal_prune_default(signal_full)