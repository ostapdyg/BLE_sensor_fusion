from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm


def __get_df(
    signals: np.ndarray, name: str, size: float, freqs: list[int], tss: np.ndarray
):
    assert freqs is not None
    data = [
        [ts, real, imag, str(freq), name, size]
        for freq, (reals, imags) in enumerate(zip(np.real(signals), np.imag(signals)))
        if freq in freqs
        for ts, real, imag in zip(tss, reals, imags)
        if not (np.isnan(real) or np.isnan(imag))
    ]
    columns = ["timestamp", "real", "imaginary", "freq", "name", "size"]
    return pd.DataFrame(data=data, columns=columns)


def get_vis_df(
    tss: np.ndarray,
    signals_data: np.ndarray,
    signals_data_pruned: np.ndarray,
    *args: tuple[np.ndarray, str],
    freqs: list[int] = None,
    n: int = None,
):
    if n is None:
        n = signals_data.shape[1]
    if freqs is None:
        freqs = list(range(signals_data.shape[0]))

    tss = tss[:n]
    data = [
        (signals_data[:, :n], "ground truth", 0.5, freqs, tss),
        (signals_data_pruned[:, :n], "measures", 1, freqs, tss),
    ] + [(signals[:, :n], name, 0.5, freqs, tss) for signals, name in args]

    with Pool(cpu_count()) as p:
        dfs = p.starmap(__get_df, data)
    return pd.concat(dfs)


def vis_signals(df: pd.DataFrame):
    return px.scatter_3d(
        df,
        x="timestamp",
        y="real",
        z="imaginary",
        color="name" if df["freq"].unique().size == 1 else "freq",
        symbol="name",
        size="size",
        height=600,
    ).update_scenes(xaxis_autorange="reversed")


def vis_signals2d(df: pd.DataFrame, kind: str = "real"):
    assert kind in df.columns
    return px.scatter(
        df,
        x="timestamp",
        y=kind,
        color="name" if df["freq"].unique().size == 1 else "freq",
        symbol="name",
        size="size",
        height=600,
    )


# def vis_fft(
#     tss: np.ndarray,
#     signals_data: np.ndarray,
#     signals_data_pruned: np.ndarray,
#     *args: tuple[np.ndarray, str],
#     n: int = None,
#     freqs: list[int] = None
# ):
#     if n is None:
#         n = signals_data.shape[1]
#     if freqs is None:
#         freqs = list(range(signals_data.shape[0]))

#     def get_df(signals: np.ndarray, name: str):
#         assert freqs is not None
#         return pd.DataFrame(
#             [
#                 {
#                     "timestamp": ts,  # Todo: add unit
#                     "abs": np.abs(v),
#                     "real": v.real,
#                     "imag": v.imag,
#                     "freq": str(freq),
#                     "name": name,
#                 }
#                 for freq, signal in enumerate(signals)
#                 for ts, v in zip(
#                     tss[:n][~np.isnan(signal)], np.fft.fft(signal[~np.isnan(signal)])
#                 )
#                 if freq in freqs
#             ]
#         )

#     df = pd.concat(
#         [
#             get_df(signals_data[:, :n], "all"),
#         ]
#         + [get_df(signals[:, :n], name) for signals, name in args]
#     )
#     return px.line(
#         df, y="abs", color="name" if len(freqs) == 1 else "freq", symbol="name"
#     )


def fft(vals, signal_tss, max_freq=None):
    sample_rate = signal_tss.shape[-1] / (signal_tss[-1] - signal_tss[0])
    fft_freqs = np.fft.fftfreq(vals.shape[-1]) * sample_rate
    fft_vals = np.fft.fft(vals) / (vals.shape[0])
    if max_freq:
        f_idxs = np.abs(fft_freqs) < max_freq
        return fft_vals[f_idxs], fft_freqs[f_idxs]
    return fft_vals, fft_freqs


def vis_fft(
    signal_vals: np.ndarray,
    signal_tss: np.ndarray,
    max_freq: int = None,
    fig=None,
    **kwargs,
):
    fft_vals, fft_freqs = fft(signal_vals, signal_tss, max_freq)
    if not max_freq:
        maxval = np.max(np.abs(fft_vals))
        max_freq = max(np.abs(fft_freqs[np.abs(fft_vals) > 0.05 * maxval])) * 1.5
        # max_freq *= 1.5
        f_idxs = np.abs(fft_freqs) < max_freq
        fft_vals = fft_vals[f_idxs]
        fft_freqs = fft_freqs[f_idxs]
    # For some reason, negative frequencies are after positive;
    fft_vals = np.concatenate([fft_vals[fft_freqs < 0.0], fft_vals[fft_freqs >= 0.0]])
    fft_freqs = np.concatenate(
        [fft_freqs[fft_freqs < 0.0], fft_freqs[fft_freqs >= 0.0]]
    )
    fig = fig or go.Figure()
    fig.add_trace(
        go.Scatter(
            y=np.abs(fft_vals),
            x=fft_freqs,
            mode="lines+markers",
            name=(kwargs["name"] if "name" in kwargs else ""),
        )
    )
    fig.update_layout(
        xaxis_title="Frequency, Hz",
        yaxis_title="Amplitude",
        title=f"Sample rate:{1/(signal_tss[1] - signal_tss[0])} Hz;",
    )
    return fig
