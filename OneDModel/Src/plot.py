import matplotlib.pyplot as plt
import numpy as np


def plot_positions(pos_true, w_m=6, h_m=5):
    # if(w_m is None):
    # w_m = np.max()

    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.set_xlim([-.05*w_m, w_m*1.05])
    ax.set_ylim([-.05*h_m, h_m*1.05])

    ax.plot(
        pos_true[:, 0],
        pos_true[:, 1],
        "-r.",
        # color="red",
        label="Real position"
    )
    ax.legend(
        # loc='upper right',
        bbox_to_anchor=(1.04, 1.0),
        framealpha=1,
    )

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.04, 1.0),
        framealpha=1,
    )
    plt.grid(True)

    return fig, ax
    # plt.show()

# dists: {name:ndarray}


def plot_dist(dist_true, dists, tss):
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.set_xlim([-0.2, tss[-1]+0.2])
    ax.set_ylim([-0.2, 6.2])

    ax.plot(
        tss,
        dist_true,
        "-r.",
        # color="red",
        label="True"
    )

    for dist_name in dists:
        ax.plot(
            tss,
            dists[dist_name],
            "-.",
            # color="red",
            label=dist_name
        )

    ax.legend(
        # loc='upper right',
        bbox_to_anchor=(1.04, 1.0),
        framealpha=1,
    )

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.04, 1.0),
        framealpha=1,
    )
    plt.grid(True)

    return fig, ax


def plot_errors(deltas, xlim=1.0, ylim=1.0):
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])

    ax.plot(
        0,
        0,
        "rx",
        # label="",
        markersize=5,
        markeredgewidth=2
    )

    ax.plot(
        deltas[:, 0],
        deltas[:, 1],
        "b.",
        # color="blue",
        # label=""
    )
    plt.grid(True)
    # ax.legend(
    # )
    return fig, ax


def plot_abserr(deltas, tss):
    fig, ax = plt.subplots()
    plot_format_axes(ax)
    # plt.axis('scaled')
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 1])
    abs_err = (deltas[:, 0]**2+deltas[:, 1]**2)**.5
    print(abs_err, tss)
    ax.plot(
        tss,
        abs_err,
        color="blue",
        # label=""
    )
    # plt.grid(True)
    # ax.legend(
    # )
    return fig, ax


def plot_particles(ax, p_old, w_old, p, w, w_m=6, h_m=5):
    N = p_old.shape[0]
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.set_xlim([-.05*w_m, w_m*1.05])
    ax.set_ylim([-.05*h_m, h_m*1.05])

    ax.scatter(p_old[:, 0], p_old[:, 1], s=2 +
               (weights**.5)*N_PARTICLES*5, alpha=0.7)
    plt.grid(True)
    return fig, ax


def plot_format_axes(ax, x=True, y=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if(x):
        ax.spines["bottom"].set_position(("data", 0))
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    else:
        ax.spines["bottom"].set_visible(False)
        ax.axes.get_xaxis().set_ticks([])

    if(y):
        ax.spines["left"].set_position(("data", 0))
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    else:
        ax.spines["left"].set_visible(False)
        ax.axes.get_yaxis().set_ticks([])

def plot_pdf(probs):
    fig, ax = plt.subplots()
    # plt.axis('scaled')
    # ax.set_xlim([-1, 100])
    # ax.set_ylim([-.05*h_m, h_m*1.05])

    ax.plot(np.arange(probs.shape[0]), probs)
    plt.grid(True)
    return fig, ax