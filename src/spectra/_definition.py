from abc import ABC, abstractmethod
import h5py
import numpy as np
from typing import Generator, Tuple, Any, Dict, List, Optional
from pathlib import Path
from scipy import stats
from src.utils._config import Movie, AnalysisConfig
from src.utils._io import DataCache, cached
from src.utils.binning import *
from statsmodels.stats.multitest import multipletests

class SpectraBase(ABC):
    """Abstract base class for spectra operations."""
    
    def __init__(
        self,
        h5_path: Path,
        roi: str,
        metric: str,
        movie: Movie,
        **kwargs: Any
    ):
        self.h5_path = h5_path
        self.roi = roi
        self.metric = metric
        self.movie = movie
        self.kwargs = kwargs
        self._validate_inputs()
    
    @abstractmethod
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        pass
    
    @abstractmethod
    def load_spectra(self) -> np.ndarray:
        """Load spectra data."""
        pass

class Spectra(SpectraBase):
    """Handle observed spectra operations."""
    
    def _validate_inputs(self) -> None:
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
        # Additional validation
    
    def load_spectra(self) -> np.ndarray:
        with h5py.File(self.h5_path, 'r') as h5f:
            return h5f[self._get_data_path()]['observed'][:]
    
    def _get_data_path(self) -> str:
        return f"data/{self.kwargs['subject_pair'][0]}_{self.kwargs['subject_pair'][1]}_{self.movie}"

class PermutedSpectra(Spectra):
    """Handle permuted spectra operations."""
    
    def load_spectra(self) -> np.ndarray:
        with h5py.File(self.h5_path, 'r') as h5f:
            return h5f[self._get_data_path()]['permutations'][self.kwargs['perm_idx']]
        
    def _get_data_path(self) -> str:
        return f"data/{self.kwargs['subject_pair'][0]}_{self.kwargs['subject_pair'][1]}_{self.movie}"
        
class SpectraProcessor:
    """Handle spectra averaging and binning operations."""
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache

    def get_min_length(self, spectra: List[np.ndarray]) -> int:
        """Find the minimum length across all spectra.
        
        Args:
            spectra: List of spectra arrays
            
        Returns:
            Minimum length found across all spectra
        """
        return min(spectrum.shape[0] for spectrum in spectra)

    def trim_spectra(
        self,
        spectra: List[np.ndarray],
        length: Optional[int] = None
    ) -> List[np.ndarray]:
        """Trim all spectra to specified or minimum length.
        
        Args:
            spectra: List of spectra arrays
            length: Target length to trim to. If None, uses minimum length.
            
        Returns:
            List of trimmed spectra arrays
        """
        if length is None:
            length = self.get_min_length(spectra)
            
        return [spectrum[:length] for spectrum in spectra]

    @cached()
    def average_spectra(
        self,
        spectra: List[np.ndarray],
        roi: str,
        metric: str,
        movies: List[str],
        permuted: bool = False,
        perm_idx: Optional[int] = None,
        length: Optional[int] = None
    ) -> np.ndarray:
        """Average spectra after trimming to consistent length.
        
        Args:
            spectra: List of 1D spectra arrays to average
            roi: Region of interest (for cache key)
            metric: Analysis metric (for cache key)
            movies: List of movies (for cache key)
            
        Returns:
            Averaged spectra array
        """ 
        # Ensure all spectra have same length
        trimmed_spectra = self.trim_spectra(spectra, length)
        
        # Stack and compute mean
        means = np.mean(trimmed_spectra, axis=0)
        stds = np.std(trimmed_spectra, axis=0)
        ranks = np.arange(1, means.shape[0] + 1) 

        return {
            'ranks': ranks,
            'means': means,
            'stds': stds
        }

    @cached()
    def bin_spectra(
        self,
        spectra: np.ndarray,
        roi: str,
        metric: str,
        movies: List[str],
        permuted: bool = False,
        perm_idx: Optional[int] = None,
        log_scale: bool = True
    ) -> np.ndarray:
        """Bin spectra into logarithmic bins and return in a format compatible with average_spectra.
        
        Args:
            spectra: Array of shape (n_samples, n_dimensions) containing spectra
            n_bins: Number of logarithmic bins
            roi: Region of interest (for cache key)
            metric: Analysis metric (for cache key)
            subjects: List of subject IDs (for cache key)
            movies: List of movie names (for cache key)
            log_scale: Whether to use logarithmic binning
            
        Returns:
            Array of binned values in a format compatible with average_spectra
        """
        if not log_scale:
            raise ValueError("Only logarithmic binning is supported")
            
        # Get the indices array
        indices = np.arange(spectra.shape[0])
        
        # Assign bin centers to each index
        bin_assignments = [assign_data_to_geometrically_spaced_bins(
            index + 1,
            density=3,
            start=1,
            stop=10_000) for index in indices]
        
        return average_bin(spectra, bin_assignments)
        # # Get unique bins and create a mapping array
        # unique_bins = np.unique(bin_assignments)
        
        # # Create a binary matrix of shape (n_unique_bins, n_points)
        # # This is like a one-hot encoding for each bin
        # bin_matrix = (bin_assignments == unique_bins[:, None])
        
        # # Use matrix multiplication to sum values in each bin
        # # (n_unique_bins, n_points) @ (n_points, n_samples).T -> (n_unique_bins, n_samples)
        # bin_sums = bin_matrix @ spectra.T
        
        # # Count number of points in each bin
        # bin_counts = bin_matrix.sum(axis=1)
        
        # # Compute means
        # # This divides each row by its bin count
        # binned_values = bin_sums / bin_counts
        
        # # Compute standard deviations using vectorized operations
        # expanded_means = (bin_matrix.T @ binned_values).T
        # squared_diff = (spectra - expanded_means) ** 2
        # sum_squared_diff = bin_matrix @ squared_diff.T
        # binned_stds = np.sqrt(sum_squared_diff / (bin_counts - 1))

        # return {
        #     'ranks': unique_bins,
        #     'means': binned_values,
        #     'stds': binned_stds
        # }
    
    
class SpectraGenerator:
    """Generate spectra for different combinations of parameters."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def yield_spectra(
        self,
        permuted: bool = False,
        perm_idx: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate spectra for all subject pairs and movies.
        
        Args:
            permuted: Whether to yield permuted spectra
            perm_idx: Index of permutation to yield (if permuted=True)
        """
        spectra_class = PermutedSpectra if permuted else Spectra
        for movie in self.config.movies:
            for i, subject1 in enumerate(self.config.subjects):
                for subject2 in self.config.subjects[i+1:]:
                    kwargs = {
                        'subject_pair': (subject1, subject2),
                    }
                    if permuted:
                        kwargs['perm_idx'] = perm_idx
                        
                    spectra = spectra_class(
                        h5_path=self.config.data_path,
                        roi=self.config.roi_names[0],
                        metric=self.config.metric,
                        movie=movie,
                        **kwargs
                    )
                    yield spectra.load_spectra()

class SpectraAnalysis:
    """Statistical analysis of spectra."""
    
    def __init__(self, observed: Dict[str, np.ndarray], permuted: List[Dict[str, np.ndarray]]):
        """
        Initialize spectra analysis.
        
        Args:
            observed: Dictionary containing 'ranks', 'means', and 'stds' for observed spectra
            permuted: List of dictionaries, each containing 'ranks', 'means', and 'stds' 
                     for each permutation
        """
        self.observed = observed
        self.permuted = permuted
        
    def compute_empirical_pvalues(self) -> Dict[str, np.ndarray]:
        """
        Compute empirical p-values for each component by comparing observed
        values against the null distribution from permutations.
        
        Returns:
            Dictionary containing:
                'pvalues': P-values for each component
                'significant': Boolean mask of significant components
                'permuted_dist': Distribution of permuted values for each component
        """
        observed_values = self.observed.means
        permuted_values = np.array([p.means for p in self.permuted])

        # For each component, count how many permuted values are >= observed
        n_permutations = len(self.permuted)
        n_exceeding = np.sum(permuted_values >= observed_values, axis=0)
        #pvalues = (n_exceeding + 1) / (n_permutations + 1)  # Add 1 for observed
        pvalues = n_exceeding / n_permutations
    
        # Bonferroni correction
        #corrected_pvalues = multipletests(pvalues, method='bonferroni')[1]
        corrected_pvalues = [min(pvalue * len(observed_values), 1) for pvalue in pvalues]

        # Compute mean and std of permuted distribution
        permuted_mean = np.mean(permuted_values, axis=0)
        permuted_std = np.std(permuted_values, axis=0)
        
        # Compute z-scores
        zscores = (observed_values - permuted_mean) / permuted_std
        
        return {
            'pvalues': pvalues,
            'zscores': zscores,
            'significant': pvalues < 0.05,  # Using 0.05 as default threshold
            'permuted_mean': permuted_mean,
            'permuted_std': permuted_std,
            'ranks': self.observed.ranks
        }

