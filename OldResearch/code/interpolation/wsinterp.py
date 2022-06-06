import numpy as np

def wsinterp(x, xp, fp, left=None, right=None):
    """One-dimensional Whittaker-Shannon interpolation.

    This uses the Whittaker-Shannon interpolation formula to interpolate the
    value of fp (array), which is defined over xp (array), at x (array or
    float).

    Returns the interpolated array with dimensions of x.

    """
    scalar = np.isscalar(x)
    if scalar:
        x = np.array(x)
        x.resize(1)
    # shape = (nxp, nx), nxp copies of x data span axis 1
    u = np.resize(x, (len(xp), len(x)))
    # Must take transpose of u for proper broadcasting with xp.
    # shape = (nx, nxp), v(xp) data spans axis 1
    v = (xp - u.T) / (xp[1] - xp[0])
    # shape = (nx, nxp), m(v) data spans axis 1
    m = fp * np.sinc(v)
    # Sum over m(v) (axis 1)
    fp_at_x = np.sum(m, axis=1)

    # Enforce left and right
    if left is None:
        left = fp[0]
    fp_at_x[x < xp[0]] = left
    if right is None:
        right = fp[-1]
    fp_at_x[x > xp[-1]] = right

    # Return a float if we got a float
    if scalar:
        return float(fp_at_x)

    return fp_at_x
