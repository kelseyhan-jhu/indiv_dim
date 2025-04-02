import numpy as np
from typing import Sequence, Optional, Tuple
from dataclasses import dataclass

N_BINS = 70
MIN_LENGTH = 1430
BINS = np.unique(np.geomspace(1, MIN_LENGTH - 1, N_BINS).astype(int))
MOVING_WINDOWS = [(BINS[i], BINS[i+1]) for i in range(len(BINS)-1)]

def assign_data_to_geometrically_spaced_bins(
    data: Sequence[float],
    /,
    *,
    n: int | None = None,
    base: int = 10,
    density: float | None = None,
    start: float | None = None,
    stop: float | None = None,
) -> np.ndarray[float]:
    """Assign each data point to a bin, where bins are spaced geometrically.

    The bins are geometrically spaced from `start` to `stop` (inclusive). The number
    of bins is determined by `n`, if provided, or `density` and `base` otherwise.
    For example, if `base == 10` and `density == 3`, there will be about 3 bins
    every decade.

    Args:
    ----
        data: Data points to be assigned to bins.
        n: The number of bins to use. Defaults to None.
        density: The approximate number of bins within each level.
        base: The base used for logarithmic binning. Defaults to 10.
        start: The start of the logarithmic scale. Defaults to `min(data)`.
        stop: The end of the logarithmic scale. Defaults to `max(data)`.

    Returns:
    -------
        The bin centers corresponding to each data point, where each bin center is the
        geometric mean of the edges corresponding to the bin.
    """
    start = min(data) if start is None else start
    stop = max(data) if stop is None else stop

    if n is None:
        if density is None:
            error = "At least one of `n` and `density` must be provided."
            raise ValueError(error)

        n_levels = (np.log(stop) - np.log(start)) / np.log(base)
        n = int(density * n_levels)

    bin_edges = np.geomspace(start, stop, num=n)
    bin_centers = np.exp(np.log(bin_edges)[:-1] + np.diff(np.log(bin_edges)) / 2)

    bin_edges[-1] = np.inf
    return bin_centers[np.digitize(data, bin_edges) - 1]

@dataclass
class BinnedSeries:
    ranks: np.ndarray
    means: np.ndarray
    stds: np.ndarray

def average_bin(
    series: np.ndarray,
    bin_assignments: np.ndarray,
    moving_windows: Optional[Sequence[Tuple[int, int]]] = None,
) -> BinnedSeries:
    
    unique_bins = np.unique(bin_assignments)
    if moving_windows is not None:
        binned_corrs = {rank: [] for rank in unique_bins}
        for n, window in enumerate(moving_windows):
            bin = bin_assignments[window[0]-1]
            binned_corrs[bin].append(series[n])
        binned_values =  np.array([np.array(binned_corrs[rank]).mean() for rank in unique_bins])
        binned_stds = np.array([np.array(binned_corrs[rank]).std() for rank in unique_bins])
    else:   
        # Create a binary matrix of shape (n_unique_bins, n_points)
        # This is like a one-hot encoding for each bin
        bin_matrix = (bin_assignments == unique_bins[:, None])
        
        # Use matrix multiplication to sum values in each bin
        # (n_unique_bins, n_points) @ (n_points, n_samples).T -> (n_unique_bins, n_samples)
        bin_sums = bin_matrix @ series.T
        
        # Count the numbenumber of points in each bin
        bin_counts = bin_matrix.sum(axis=1)
        
        # Compute means
        # This divides each row by its bin count
        binned_values = bin_sums / bin_counts
        
        # Compute standard deviations using vectorized operations
        expanded_means = (bin_matrix.T @ binned_values).T
        squared_diff = (series - expanded_means) ** 2
        sum_squared_diff = (bin_matrix @ squared_diff.T) / bin_counts
        binned_stds = np.sqrt(sum_squared_diff)
        #sum_squared_diff = bin_matrix @ squared_diff.T
        #binned_stds = np.sqrt(sum_squared_diff / (bin_counts - 1))
    return BinnedSeries(ranks=unique_bins, means=binned_values, stds=binned_stds)