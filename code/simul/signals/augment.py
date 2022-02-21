import numpy as np

from simul.parameters import Parameters


def signal_add_noise(signals:np.ndarray, val:float = 1, noise_type:str = "normal") -> np.ndarray:
    if(noise_type == "normal"):
        return signals + np.random.normal(0, val, signals.shape) + 1j*np.random.normal(0, val, signals.shape)
    if(noise_type == "uniform"):
        return signals + np.random.uniform(-val, val, signals.shape) + 1j*np.random.uniform(-val, val, signals.shape)


def signal_drop_vals(signals:np.ndarray, noise_type:str = "default") -> np.ndarray:
    
    pass