import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_RMSE(df: pd.DataFrame, per_freq: bool = False):
    residuals = df[(df["name"] != "measures") & (df["name"] != "ground truth")]
    ground_truth = df[df["name"] == "ground truth"]
    for freq in tqdm(residuals["freq"].unique()):
        idx = residuals["freq"] == freq
        residuals.loc[idx, "real"] -= residuals.loc[idx, "timestamp"].map(
            ground_truth[ground_truth["freq"] == freq].set_index("timestamp")["real"]
        )
        residuals.loc[idx, "imaginary"] -= residuals.loc[idx, "timestamp"].map(
            ground_truth[ground_truth["freq"] == freq].set_index("timestamp")[
                "imaginary"
            ]
        )
    rmse = residuals.groupby("name").apply(
        lambda df: df.groupby("freq").apply(
            lambda x: np.sqrt((x["real"] ** 2 + x["imaginary"] ** 2).mean())
        )
    )
    if per_freq:
        return rmse
    return pd.DataFrame(rmse.sum(1), columns=["RMSE"]).T
