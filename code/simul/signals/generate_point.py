import numpy as np
from simul.parameters import Parameters
from simul.walls.wall import SignalPath

def generate_point(p: Parameters, ts: float) -> tuple[np.ndarray, np.ndarray]:
    key_shift_m = (p.vel / 3600) * 1000 * ts
    key_x = p.start_pos - key_shift_m

    # dist = np.array([[d_func(key_x, ts) for d_func in p.dist_funcs]])
    # Serializable walls
    dist = np.array([[SignalPath.from_dict(path_dict).path_length(key_x, ts) for path_dict in p.signal_paths]])
    ampl_coeff = np.expand_dims(
        np.array(p.scenario_matrix).repeat(dist.shape[0]), axis=0
    )
    ampl_coeff = ampl_coeff + p.scenario_noise * np.random.uniform(
        0, 1, ampl_coeff.shape
    )

    return (dist, ampl_coeff)
