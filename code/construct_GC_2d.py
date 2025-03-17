import numpy as np

def construct_GC_2d_general(cut, mlocs, ylocs, Nx=None):
    """
    Construct the Gaspari and Cohn localization matrix for a 2D field.

    Parameters:
        cut (float): Localization cutoff distance.
        mlocs (array of shape (nstates,2)): 2D coordinates of the model states [(x1, y1), (x2, y2), ...].
        ylocs (array of shape (nobs,2)): 2D coordinates of the observations [[x1, y1], [x2, y2], ...].
        Nx (int, optional): Number of grid points in each direction.

    Returns:
        np.ndarray: Localization matrix of shape (len(ylocs), len(mlocs)).
    """
    ylocs = ylocs[:, np.newaxis, :]  # Shape (nobs, 1, 2)
    mlocs = mlocs[np.newaxis, :, :]  # Shape (1, nstates, 2)

    # Compute distances
    dist = np.linalg.norm((mlocs - ylocs + Nx // 2) % Nx - Nx // 2, axis=2)

    # Normalize distances
    r = dist / (0.5 * cut)

    # Compute localization function
    V = np.zeros_like(dist)

    mask2 = (0.5 * cut <= dist) & (dist < cut)
    mask3 = (dist < 0.5 * cut)

    V[mask2] = (
        r[mask2]**5 / 12.0 - r[mask2]**4 / 2.0 + r[mask2]**3 * 5.0 / 8.0
        + r[mask2]**2 * 5.0 / 3.0 - 5.0 * r[mask2] + 4.0 - 2.0 / (3.0 * r[mask2])
    )
    
    V[mask3] = (
        -r[mask3]**5 * 0.25 + r[mask3]**4 / 2.0 + r[mask3]**3 * 5.0 / 8.0 
        - r[mask3]**2 * 5.0 / 3.0 + 1.0
    )

    return V