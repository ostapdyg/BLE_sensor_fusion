import matplotlib.pyplot as plt
import numpy as np

def plot_positions(pos_true, pos_predicted, anchors, w_m=6, h_m=5):
    # if(w_m is None):
        # w_m = np.max()
    
    fig, ax = plt.subplots()
    plt.axis('scaled')
    ax.set_xlim([-.05*w_m, w_m*1.05])
    ax.set_ylim([-.05*h_m, h_m*1.05])

    ax.plot(
        [*anchors[:,0], 0], 
        [*anchors[:,1], 0], 
        "-o",
        color="black",
        mfc='white',
        linewidth=3,
        markersize=12,
        markeredgewidth=3,

        label="Anchors"
    )
    
    ax.plot(
        pos_true[:,0],
        pos_true[:,1],
        "-r.",
        # color="red",
        label="Real position"
    )

    ax.plot(
        pos_predicted[:,0],
        pos_predicted[:,1],
        "-bx",
        label="Predicted position",
        markersize=5,
        markeredgewidth=2
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

def plot_errors(deltas, xlim = 1.0, ylim = 1.0):
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
        deltas[:,0],
        deltas[:,1],
        "b.",
        # color="blue",
        # label=""
    )
    plt.grid(True)
    # ax.legend(
    # )
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
