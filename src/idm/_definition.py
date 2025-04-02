import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from abc import ABC, abstractmethod
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from src.utils._config import Movie, AnalysisConfig, Subjects
from src.utils._io import DataCache, cached
from src.utils.binning import *
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity

#BASE_EXCLUDED = ["sub-NSD113", "sub-NSD114", "sub-NSD115", "sub-NSD155"]
#ALL_EXCLUDED = ["sub-NSD107"] + BASE_EXCLUDED
ALL_EXCLUDED = ["sub-NSD107"]

def get_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle values from a matrix, excluding diagonal."""
    mask = np.triu_indices(matrix.shape[0], k=1)
    return matrix[mask]

def get_filtered_upper_triangle(matrix: np.ndarray, subject_idx: np.ndarray) -> np.ndarray:
    """
    Get upper triangle values while excluding self-similarity values from resampling.
    
    Args:
        matrix: Input matrix
        subject_idx: Array of subject indices used in bootstrapping
    """
    n = matrix.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    values = matrix[upper_idx]
    
    # Create mask for self-similarity values
    mask = np.zeros_like(values, dtype=bool)
    for i, (row, col) in enumerate(zip(*upper_idx)):
        # Check if these positions represent same subject due to bootstrap resampling
        if subject_idx[row] == subject_idx[col]:
            mask[i] = True
    
    # Return only non-self-similarity values
    return values[~mask]

@dataclass
class BinStrategy:
    """Define binning strategy for IDM correlations."""
    type: str  # 'decade' or 'logarithmic'
    density: Optional[int] = 3
    start: Optional[int] = 1
    stop: Optional[int] = 10_000
    
    def get_bins(self, length) -> List[Tuple[int, int]]:
        if self.type == 'decade':
            return [(1, 10), (10, 100), (100, 1000)]
        elif self.type == 'logarithmic':
            indices = np.arange(length)
            bin_assignments = [
                assign_data_to_geometrically_spaced_bins(
                    index + 1,
                    density=self.density,
                    start=self.start,
                    stop=self.stop
                ) for index in indices
            ]
            return bin_assignments
            #return [(int(unique_bins[i]), int(unique_bins[i+1])) 
            #       for i in range(len(unique_bins)-1)]
        raise ValueError(f"Unknown bin strategy: {self.type}")

class AnatomicalMatrix:
    def __init__(
        self,
        h5_path: Path,
        roi: str,
        metric: str,
        movie: str,
        matrix_type: str = 'anatomical'
    ):
        self.h5_path = h5_path
        self.roi = roi
        self.metric = metric
        self.movie = movie
        if matrix_type not in ['anatomical', 'isc']:
            raise ValueError("matrix_type must be either 'anatomical' or 'isc'")
        self.matrix_type = matrix_type
        
    def get_aligned_matrix(self, target_subjects: List[str]) -> np.ndarray:
        """Get matrix aligned to target subject list."""
        n_subjects = len(target_subjects)
        matrix = np.zeros((n_subjects, n_subjects))
        
        with h5py.File(self.h5_path, 'r') as h5f:
            for i, i_subject in enumerate(target_subjects):
                for j, j_subject in enumerate(target_subjects[i+1:], i+1):
                    key = f"data/{i_subject}_{j_subject}_{self.movie}"
                    if key in h5f:
                        if type(h5f[key]['observed'][()]) == np.ndarray:
                            matrix[i, j] = h5f[key]['observed'][()].mean()
                        else:
                            matrix[i, j] = h5f[key]['observed'][()]                        
        return matrix
 
class MotionMatrix:
    def __init__(self, motion_dir: Path, movie: str):
        self.motion_dir = motion_dir
        self.movie = movie
        
    def get_aligned_matrix(self, motion_type: str, target_subjects: List[str]) -> np.ndarray:
        """Load and align motion matrix to target subject list."""
        if motion_type not in ['ISC', 'AnnaK']:
            raise ValueError("motion_type must be either 'ISC' or 'AnnaK'")
            
        motion_path = self.motion_dir / f"motion{motion_type}_matrix_{self.movie}.npy"
        full_matrix = np.load(motion_path)
        
        # Get indices for alignment
        full_subjects = [s for s in Subjects.get_analysis_subjects() if s not in ["sub-NSD107"]]
        target_indices = [full_subjects.index(s) for s in target_subjects]
        
        # Extract relevant rows/columns
        return full_matrix[target_indices][:, target_indices]

class MemoryMatrix:
    def __init__(self, memory_dir: Path, movie: str):
        self.memory_dir = memory_dir
        self.movie = movie
    
    def create_use_rdm(embs, movie, motion=True):
        capitalized_movie = movie.capitalize()
        vector = embs[capitalized_movie]
        use_rdm = cosine_similarity(vector, vector)
        rows_cols_to_keep = list(range(use_rdm.shape[0]))
        rows_cols_to_remove = [9, 10, 11, 41]
        for index in sorted(rows_cols_to_remove, reverse=True):
            del rows_cols_to_keep[index]
        use_rdm = use_rdm[np.ix_(rows_cols_to_keep, rows_cols_to_keep)]
        # # if motion:
        # #     # motion = np.load(f"../../finn/motionAnnaK_matrix_{movie}.npy")
        # #     # rows_cols_to_keep = list(range(motion.shape[0]))
        # #     # rows_cols_to_remove = [4, 9, 10, 11, 41]
        # #     # for index in sorted(rows_cols_to_remove, reverse=True):
        # #     #     del rows_cols_to_keep[index]
        # #     # motion = motion[np.ix_(rows_cols_to_keep, rows_cols_to_keep)]
        # #     # motion_triu = motion[np.triu_indices(38, k=1)]
        # #     # use_rdm_triu_normalized = zscore(use_rdm_triu)
        # #     # motion_triu_normalized = zscore(motion_triu)
        # #     # use_processed = use_rdm_triu_normalized - motion_triu_normalized
        # #     use_processed = motion_normalization(movie, [4, 9, 10, 11, 41], use_rdm)
        # #     use_rdm_triu = use_processed[np.triu_indices(38, k=1)]
        # else:
        use_rdm_triu = use_rdm[np.triu_indices(38, k=1)]
        return use_rdm_triu
    
class IDMGenerator:
    def __init__(
        self,
        cache: DataCache,
        path: Path,
        subjects: List[str],
        movie: str
    ):
        self.cache = cache
        self.path = path
        self.subjects = subjects
        self.movie = movie
        
    def load_spectra(self) -> np.ndarray:
        """Load all spectra for the movie."""
        with h5py.File(self.path, 'r') as h5f:
            n_subjects = len(self.subjects)
            
            # Initialize array to store all spectra
            all_spectra = np.zeros((n_subjects, n_subjects, len(MOVING_WINDOWS)))
            
            # Load spectra for each subject pair
            for i, i_subject in enumerate(self.subjects):
                for j, j_subject in enumerate(self.subjects[i+1:], i+1):
                    spectrum = h5f[f"data/{i_subject}_{j_subject}_{self.movie}"]["observed"][:]
                    for n_window, windows in enumerate(MOVING_WINDOWS):
                        all_spectra[i, j, n_window] = spectrum[windows[0]:windows[1]].mean()
                        all_spectra[j, i, n_window] = all_spectra[i, j, n_window]
                    
            return all_spectra
        
    def load_permuted_spectra(self) -> np.ndarray:
        """Load all spectra for the movie."""
        with h5py.File(self.path, 'r') as h5f:
            n_subjects = len(self.subjects)
            
            # Initialize array to store all spectra
            all_spectra = np.zeros((n_subjects, n_subjects, len(MOVING_WINDOWS)))
            
            perm_i = np.random.permutation(self.subjects)
            perm_j = np.random.permutation(self.subjects)

            # Load spectra for each subject pair
            for i, i_subject in enumerate(perm_i):
                for j, j_subject in enumerate(perm_j):
                    if i_subject != j_subject:
                        try:
                            spectrum = h5f[f"data/{i_subject}_{j_subject}_{self.movie}"]["observed"][:]
                        except:
                            spectrum = h5f[f"data/{j_subject}_{i_subject}_{self.movie}"]["observed"][:]
                        for n_window, windows in enumerate(MOVING_WINDOWS):
                            all_spectra[i, j, n_window] = spectrum[windows[0]:windows[1]].mean()
                            all_spectra[j, i, n_window] = all_spectra[i, j, n_window]
            return all_spectra
    
    @cached(lambda roi, metric, movie: f"idm_{roi}_{metric}_{movie}")  
    def generate_idm_series(self, roi: str, metric: str, movie: str) -> np.ndarray:
        """Generate IDM for each rank."""
        spectra = self.load_spectra()
        return np.moveaxis(spectra, -1, 0)  # reshape to (ranks, n_subjects, n_subjects)

    @cached(lambda roi, metric, movie, perm_idx: f"idm_{roi}_{metric}_{movie}_perm_{str(perm_idx)}")  
    def permute_idm_series(self, roi: str, metric: str, movie: str, perm_idx: int) -> np.ndarray:
        """Permute IDM for each rank."""
        spectra = self.load_permuted_spectra()
        return np.moveaxis(spectra, -1, 0)  # reshape to (ranks, n_subjects, n_subjects)

class IDMAnalysis:
    def __init__(self, cache: DataCache, bin_strategy: BinStrategy):
        self.cache = cache
        self.bin_strategy = bin_strategy

    def _partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        # Validate input shapes
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        z = np.asarray(z).flatten()

        # Create DataFrame with original data
        data = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z
        })
        
        # Convert all data to ranks first, as pingouin does
        data_ranked = data.rank()
        
        # Calculate covariance matrix of ranked data
        V = data_ranked.cov()
        
        # Calculate inverse covariance matrix
        Vi = np.linalg.pinv(V, hermitian=True)
        Vi_diag = Vi.diagonal()
        D = np.diag(np.sqrt(1 / Vi_diag))
        pcor = -1 * (D @ Vi @ D)
        
        # Calculate residuals using matrix algebra on ranked data
        Z = sm.add_constant(data_ranked['z'])
        Hat = Z @ np.linalg.pinv(Z)
        I = np.eye(len(x))
        M = I - Hat
        
        res_x = M @ data_ranked['x']
        res_y = M @ data_ranked['y']
        
        return pcor[0, 1], res_x, res_y
    
        # """Compute partial correlation between x and y controlling for z."""    
        # x_rank = stats.rankdata(x)
        # y_rank = stats.rankdata(y)
        # z_rank = stats.rankdata(z)
        
        # # Standardize ranks
        # def standardize(arr):
        #     return (arr - np.mean(arr)) / np.std(arr)
        
        # x_std = standardize(x_rank)
        # y_std = standardize(y_rank)
        # z_std = standardize(z_rank)
        
        # # Fit on standardized ranks
        # Z = sm.add_constant(z_std)
        # model_x = sm.OLS(x_std, Z).fit()
        # model_y = sm.OLS(y_std, Z).fit()
        
        # res_x = model_x.resid
        # res_y = model_y.resid
        
        # corr = stats.spearmanr(res_x, res_y)[0]
        
        # return corr, res_x, res_y

    def _fully_partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xz: np.ndarray,
        yz: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        # Validate input shapes
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        xz = np.asarray(xz).flatten()
        yz = np.asarray(yz).flatten()

        data_x = pd.DataFrame({
            'x': x,
            'xz': xz
        })
        
        data_y = pd.DataFrame({
            'y': y,
            'yz': yz
        })
        
        # Convert all data to ranks
        data_x_ranked = data_x.rank()
        data_y_ranked = data_y.rank()
        
        # Calculate residuals for x using matrix algebra
        Zx = sm.add_constant(data_x_ranked['xz'])
        Hat_x = Zx @ np.linalg.pinv(Zx)
        I = np.eye(len(x))
        Mx = I - Hat_x
        res_x = Mx @ data_x_ranked['x']
        
        # Calculate residuals for y using matrix algebra
        Zy = sm.add_constant(data_y_ranked['yz'])
        Hat_y = Zy @ np.linalg.pinv(Zy)
        My = I - Hat_y
        res_y = My @ data_y_ranked['y']
        
        # Calculate correlation between residuals
        correlation = stats.spearmanr(res_x, res_y)[0]
        
        return correlation, res_x, res_y

    
    def compute_correlation(
        self,
        idm1: np.ndarray,
        idm2: np.ndarray,
        nuisance: Optional[np.ndarray] = None,
        subject_idx: Optional[np.ndarray] = None
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """Compute correlation between two IDMs with optional nuisance variable."""
        if subject_idx is not None:
            idm1_upper = get_filtered_upper_triangle(idm1, subject_idx)
            idm2_upper = get_filtered_upper_triangle(idm2, subject_idx)
        else:
            idm1_upper = get_upper_triangle(idm1)
            idm2_upper = get_upper_triangle(idm2)
        
        if nuisance is None:
            return stats.spearmanr(idm1_upper, idm2_upper)[0]
        
        if subject_idx is not None:
            nuisance_upper = get_filtered_upper_triangle(nuisance, subject_idx)
        else:
            nuisance_upper = get_upper_triangle(nuisance)
        return self._partial_correlation(idm1_upper, idm2_upper, nuisance_upper)

    def compute_correlation_series(
        self,
        idm_series1: np.ndarray,
        idm_series2: np.ndarray,
        nuisance: Optional[np.ndarray] = None,
        subject_idx: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]]:
        """Compute correlations for each rank bin."""
        n_bins = len(idm_series1)
        correlations = np.zeros(n_bins)
        residuals_x = []
        residuals_y = []
        
        for t in range(n_bins):
            if nuisance is None:
                correlations[t] = self.compute_correlation(idm_series1[t], idm_series2[t], nuisance = None, subject_idx = subject_idx)
            else:
                corr, res_x, res_y = self.compute_correlation(
                    idm_series1[t], idm_series2[t], nuisance = nuisance, subject_idx = subject_idx
                )
                correlations[t] = corr
                residuals_x.append(res_x)
                residuals_y.append(res_y)
        
        if nuisance is None:
            return correlations
        return correlations, residuals_x, residuals_y

    def bin_correlations(
        self,
        correlations: np.ndarray,
        length: int
    ) -> Dict[Tuple[int, int], float]:
        """Bin correlation series."""
        bin_assignments = self.bin_strategy.get_bins(length=length)
        return average_bin(correlations, bin_assignments, MOVING_WINDOWS)

    def IDM_correlations_even_odd(
        self,
        idm_series: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IDMs for even and odd movie sets."""
        min_length = min([idm.shape[0] for idm in idm_series.values()])
        movies = list(idm_series.keys())
        even_movies = movies[::2]
        odd_movies = movies[1::2]
        
        even_idms = np.nanmean([idm_series[movie][:min_length] for movie in even_movies], axis=0)
        odd_idms = np.nanmean([idm_series[movie][:min_length] for movie in odd_movies], axis=0)
        
        return even_idms, odd_idms

    def IDM_correlations_average_pairs(
        self,
        idm_series: Dict[str, np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[str, str]]]:
        """Return list of IDM pairs and their corresponding movies."""
        min_length = min([idm.shape[0] for idm in idm_series.values()])
        movies = list(idm_series.keys())
        
        even_pairs = [(movies[i], movies[j]) 
                     for i in range(0, len(movies), 2)
                     for j in range(i+2, len(movies), 2)]
        odd_pairs = [(movies[i], movies[j]) 
                    for i in range(1, len(movies), 2)
                    for j in range(i+2, len(movies), 2)]
        
        pairs = []
        for m1, m2 in even_pairs + odd_pairs:
            pairs.append((
                idm_series[m1][:min_length],
                idm_series[m2][:min_length],
                (m1, m2)
            ))
        return pairs

def create_nuisance_matrices(args, config):
    """Create and align nuisance matrices based on analysis scenario."""
    matrices = {}
    
    # Get base subject list (excluding all specified subjects)
    base_subjects = [s for s in Subjects.get_analysis_subjects() 
                    if s not in ALL_EXCLUDED]
    
    movies = ["iteration", "defeat", "growth", "lemonade"]
    # Handle anatomical/ISC matrix
    if args.anatomical or args.isc:
        matrix_type = 'anatomical' if args.anatomical else 'isc'
        matrix_dir = args.anatomical_dir if args.anatomical else args.isc_dir
        if not matrix_dir:
            raise ValueError(f"{matrix_type} directory must be provided when using {matrix_type} correction")
            
        matrices['anatomical'] = {}
        for movie in config.movies:
            matrices['anatomical'][movie] = AnatomicalMatrix(
                Path(matrix_dir),
                config.roi_names[0],
                config.metric,
                movie,
                matrix_type
            ).get_aligned_matrix(base_subjects)
        
    # Handle motion matrices
    if args.motion:
        if not args.motion_dir:
            raise ValueError("Motion directory required when using motion correction")
            

        motion_type = args.motion_type if hasattr(args, 'motion_type') else 'ISC'
        if type(motion_type) == list:
            for motion in motion_type:
                matrices[f'motion_{motion}'] = {}
                for movie in movies:
                    matrices[f'motion_{motion}'][movie] = MotionMatrix(
                        Path(args.motion_dir),
                        movie
                    ).get_aligned_matrix(motion, base_subjects)
        else:
            matrices['motion'] = {}
            for movie in movies:
                matrices['motion'][movie] = MotionMatrix(
                    Path(args.motion_dir),
                    movie
                ).get_aligned_matrix(motion_type, base_subjects)

    return matrices, base_subjects

