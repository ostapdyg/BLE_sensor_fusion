import numpy as np

ANCHOR_COORDS = np.array([
    (0,0,2),
    (6, 0, 2),
    (6, 5, 2),
    (0, 5, 2),
])

TARGET_SPEED = 1 #m/s
TARGET_START = np.array([1, 4, 1])
TARGET_PATH = np.array([
    (4, 0, 0),
    (0, -3, 0),
    (-4, 3, 0),
    (3, 0, 0),
])

TARGET_TURNS = {
    12:(np.pi*1.5),
    21:(np.pi + 0.9)
}


BASE_FREQ = 2.4e9 #Hz
FREQ_STEP = 2e6 #Hz
FREQ_NUM = 40

FREQS = np.arange(BASE_FREQ, BASE_FREQ+FREQ_STEP*FREQ_NUM-1, FREQ_STEP)
C_SPEED = 299_792_458

SIGNAL_STD = 0.02
# WALK_STD = 0.2

def simul_get_pos(t)->np.ndarray:
    path_traveled = TARGET_SPEED*t
    res = np.copy(TARGET_START)
    for path_vector in TARGET_PATH:
        vector_len = np.sqrt(path_vector.dot(path_vector))
        if(path_traveled <= vector_len):
            return res + path_traveled*(path_vector/vector_len)
        res += path_vector
        path_traveled -= vector_len
    return np.array([1., 3., 0.])
    pass

# def imu_get_turn(t)->np.ndarray:
#     path_traveled = TARGET_SPEED*t
#     res = np.copy(TARGET_START)
#     for path_vector in TARGET_PATH:
#         vector_len = np.sqrt(path_vector.dot(path_vector))
#         if(path_traveled vector_len):
#             return res + path_traveled*(path_vector/vector_len)
#         res += path_vector
#         path_traveled -= vector_len


# def simul_signal_shift(anchor, target, freq)->complex:
#     virt_targets = np.array([target])
#     res = 0+0j
#     for virt_target in virt_targets:
#         path_vector = virt_target - anchor
#         path_len = np.sqrt(path_vector.dot(path_vector))
#         omega = 2*np.pi*freq
#         a = C_SPEED / (2 * path_len * omega)  # amplitude from distance
#         phi = (path_len * omega) / C_SPEED  # phase from distance
#         res += a * np.exp(-1j * phi)

#     return res



def simul_signals_shift_full(anchor, target)->np.ndarray:
    virt_targets = [
        (target, 1),
        (np.array([target[0], 10 - target[1], target[2]]), 0.4),
        (np.array([target[0],  - target[1], target[2]]), 0.4),
        (np.array([12 - target[0], target[1], target[2]]), 0.4),
        (np.array([-target[0], target[1], target[2]]), 0.4),
        (np.array([target[0], target[1], -target[2]]), 0.4),

        ]
    signal_v = np.zeros(FREQ_NUM, dtype=np.complex128)
    for virt_target, reflection_coeff in virt_targets:
        path_vector = virt_target - anchor
        path_len = np.sqrt(path_vector.dot(path_vector))

        # print(f"path_len:{path_len}", end="\n")

        omega_v = 2*np.pi*FREQS
        # print(omega_v.shape)
        a_v = C_SPEED / (2 * path_len * omega_v)  # amplitude from distance
        # a_v = 1  # amplitude from distance

        phi_v = (path_len * omega_v) / C_SPEED  # phase from distance
        signal_v += reflection_coeff * a_v * (np.exp(-1j * phi_v) + np.random.normal(0, SIGNAL_STD)+1j*np.random.normal(0, SIGNAL_STD))

    return signal_v

    pass

# Returns fft_data, fft_freqs



# def get_plot(xss):
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=ANCHOR_COORDS[:,0], 
#         y=ANCHOR_COORDS[:,1], 
#         name="Anchors", 
#         mode = "markers",
#         marker_size=16,
#         marker_symbol='circle',
#         marker_color = 'white',
#         marker = dict(line = dict(color='green', width = 2))
#     ))


#     fig.add_trace(go.Scatter(x=xss[:,0], y=xss[:,1], name="Real position"))

#     fig.update_layout(
#         # title="Estimated ",
#         xaxis_title="x, m",
#         yaxis_title="y, m",
#         legend_title="Legend Title",

#     )

#     return fig

