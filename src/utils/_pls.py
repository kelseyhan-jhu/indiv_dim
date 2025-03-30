import logging
import os
import sys
import yaml
import nibabel as nib
import numpy as np
import torch
#from utilities.computation import svd
#from bonner.computation.decomposition._svd import svd


def load_subject_movie_data(subj, movie, roi_name, data_path):
    """
    Load fMRI data for a specific subject and movie.

    Parameters:
        subj (str): Subject identifier.
        movie (str): Movie identifier.
        roi_name (str): Region of interest name.
        data_path (str): Base path to the data.
        movie_trs (dict): Dictionary of TR start and end times for movies.

    Returns:
        np.ndarray or None: Loaded data array or None if an error occurs.
    """
    movie_trs = {
    'growth': (1, 505),
    'lemonade': (1, 449),
    'defeat': (2, 480),
    'iteration': (2, 748)
    }
    try:
        if "aal" in roi_name:
            filename = f"{subj}/func/{subj}_task-{movie}_space-MNI152NLin2009cAsym_desc-{roi_name}_bold.nii.gz"
        else:
            filename = f"{subj}/func/{subj}_task-{movie}_space-MNI152NLin2009cAsym_desc-bptf-{roi_name}_bold.nii.gz"
        full_path = os.path.join(data_path, filename)
        img = nib.load(full_path)
        if movie not in movie_trs.keys():
            data_array = img.get_fdata()
        else:
            tr_start, tr_end = movie_trs[movie]
            clipped_img = img.slicer[:, :, :, tr_start:tr_end]
            data_array = clipped_img.get_fdata()
        return data_array
    except Exception as e:
        logging.error(f"Error loading data for {subj}, movie {movie}: {e}")
        return None

def get_common_indices(sub_xarr, sub_pair, movies, n_downsample=0):
    """
    Determine common non-zero voxel indices across subjects and movies.

    Parameters:
        sub_xarr (dict): Nested dictionary of subject and movie data arrays.
        sub_pair (tuple): Pair of subjects being analyzed.
        movies (list): List of movies.

    Returns:
        np.ndarray: Array of common non-zero voxel indices.
    """
    all_nonzeros = {subj: {mov: None for mov in movies} for subj in sub_pair}
    for subj in sub_pair:
        for movie in movies:
            data_array = sub_xarr[subj][movie]
            nonzeros = np.argwhere(np.any(data_array, axis=3))
            all_nonzeros[subj][movie] = set(map(tuple, nonzeros))
    # Compute the intersection of non-zero indices
    common_nonzeros = set.intersection(*(all_nonzeros[subj][mov] for subj in sub_pair for mov in movies))
    
    # Downsample the common non-zero indices if necessary
    if n_downsample > 0:
        common_nonzeros = np.array(list(common_nonzeros))
        n_voxels = common_nonzeros.shape[0]
        if n_downsample > n_voxels:
            n_downsample = n_voxels
        indices = np.random.choice(n_voxels, size=n_downsample, replace=False)
        common_nonzeros = common_nonzeros[indices]
    
    nonzeros_common = np.array(list(common_nonzeros))
    return nonzeros_common

def extract_data(sub_xarr, sub_pair, movies, nonzeros_common):
    """
    Extract data using the common non-zero voxel indices.

    Parameters:
        sub_xarr (dict): Nested dictionary of subject and movie data arrays.
        sub_pair (tuple): Pair of subjects being analyzed.
        movies (list): List of movies.
        nonzeros_common (np.ndarray): Array of common non-zero voxel indices.

    Returns:
        dict: Updated sub_xarr with extracted data.
    """
    for subj in sub_pair:
        for movie in movies:
            data_array = sub_xarr[subj][movie]
            extracted_data = data_array[
                nonzeros_common[:, 0],
                nonzeros_common[:, 1],
                nonzeros_common[:, 2],
                :
            ]
            mean_time = extracted_data.mean(axis=0)
            std_time = extracted_data.std(axis=0)
            extracted_data = (extracted_data - mean_time) / std_time
            sub_xarr[subj][movie] = extracted_data
    return sub_xarr


def load_config(config_path='./slurm_jobs/config.yaml'):
    """
    Load configuration parameters from a YAML file.

    Parameters:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration parameters.
    """
    print("current directory is", os.getcwd())
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging():
    """
    Set up logging to a file and console.

    Parameters:
        log_file (str): Filename for logging output.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def svd_flip(*, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs_cols = torch.argmax(torch.abs(u), dim=-2)
    match u.ndim:
        case 3:
            signs = torch.stack(
                [
                    torch.sign(u[i_batch, max_abs_cols[i_batch, :], range(u.shape[-1])])
                    for i_batch in range(u.shape[0])
                ],
                dim=0,
            )
        case 2:
            signs = torch.sign(u[..., max_abs_cols, range(u.shape[-1])])

    u *= signs.unsqueeze(-2)
    v *= signs.unsqueeze(-1)

    return u, v


def svd(
    x: torch.Tensor,
    *,
    truncated: bool,
    n_components: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if truncated:
        torch.manual_seed(seed)
        u, s, v = torch.pca_lowrank(x, center=False, q=n_components)
        v_h = v.transpose(-2, -1)
    else:
        u, s, v_h = torch.linalg.svd(x, full_matrices=False)
    u, v_h = svd_flip(u=u, v=v_h)
    return u, s, v_h.transpose(-2, -1)




class PLSSVD:
    def __init__(self) -> None:
        self.left_mean: np.ndarray
        self.right_mean: np.ndarray
        self.left_singular_vectors: np.ndarray
        self.right_singular_vectors: np.ndarray

    def fit(self, /, x: np.ndarray, y: np.ndarray, random_state) -> None:
        self.left_mean = x.mean(axis=-2)
        self.right_mean = y.mean(axis=-2)
        self.left_std = x.std(axis=-2)
        self.right_std = y.std(axis=-2)

        x_centered = (x - self.left_mean) / self.left_std
        y_centered = (y - self.right_mean) / self.right_std

        n_stimuli = x.shape[-2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cross_covariance = (np.swapaxes(x_centered, -1, -2) @ y_centered) / (
            n_stimuli - 1
        )

        (
            self.left_singular_vectors,
            self.singular_values,
            self.right_singular_vectors,
        ) = svd(
            torch.from_numpy(cross_covariance).to(device),
            n_components=min([*x.shape, *y.shape]),
            truncated=True,
            seed=random_state,
        )

        self.left_singular_vectors = self.left_singular_vectors.cpu().numpy()
        self.singular_values = self.singular_values.cpu().numpy()
        self.right_singular_vectors = self.right_singular_vectors.cpu().numpy()

    def transform(self, /, z: np.ndarray, *, direction: str) -> np.ndarray:
        match direction:
            case "left":
                return (z - self.left_mean) / self.left_std @ self.left_singular_vectors
            case "right":
                return (z - self.right_mean) / self.right_std @ self.right_singular_vectors
            case _:
                raise ValueError("direction must be 'left' or 'right'")
            
    def inverse_transform(self, /, z: np.ndarray, *, direction: str) -> np.ndarray:
        match direction:
            case "left":
                return z @ self.left_singular_vectors.T
            case "right":
                return z @ self.right_singular_vectors.T
            case _:
                raise ValueError("direction must be 'left' or 'right'")  
# class PLSSVD:
#     def __init__(self) -> None:
#         self.left_mean: np.ndarray
#         self.right_mean: np.ndarray
#         self.left_singular_vectors: np.ndarray
#         self.right_singular_vectors: np.ndarray

#     def fit(self, /, x: np.ndarray, y: np.ndarray, random_state) -> None:
#         # Calculate statistics 
#         self.left_mean = x.mean(axis=-2)
#         self.right_mean = y.mean(axis=-2)
#         self.left_std = x.std(axis=-2)
#         self.right_std = y.std(axis=-2)

#         # Center and normalize
#         x_centered = (x - self.left_mean) / self.left_std
#         y_centered = (y - self.right_mean) / self.right_std
#         n_stimuli = x.shape[-2]
        
#         # Calculate cross-covariance
#         cross_covariance = (np.swapaxes(x_centered, -1, -2) @ y_centered) / (n_stimuli - 1)
        
#         # Perform SVD directly
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         n_components = min([*x.shape, *y.shape])
        
#         if torch.cuda.is_available():
#             # GPU path
#             cross_covariance_tensor = torch.from_numpy(cross_covariance).to(device)
#             u, s, v_t = torch.linalg.svd(cross_covariance_tensor, full_matrices=False)
#             v = v_t.transpose(-2, -1)
            
#             # SVD flip
#             max_abs_cols = torch.argmax(torch.abs(u), dim=-2)
#             if u.ndim == 3:
#                 signs = torch.stack(
#                     [torch.sign(u[i, max_abs_cols[i, :], range(u.shape[-1])]) for i in range(u.shape[0])],
#                     dim=0,
#                 )
#             else:
#                 signs = torch.sign(u[max_abs_cols, range(u.shape[-1])])
            
#             u *= signs.unsqueeze(-2)
#             v *= signs.unsqueeze(-1)
            
#             # Convert results to NumPy
#             self.left_singular_vectors = u[..., :n_components].cpu().numpy()
#             self.singular_values = s[..., :n_components].cpu().numpy()
#             self.right_singular_vectors = v[..., :n_components].cpu().numpy()
#         else:
#             # CPU path
#             u, s, v_t = np.linalg.svd(cross_covariance, full_matrices=False)
#             v = v_t.transpose(-1, -2)
            
#             # SVD flip
#             max_abs_cols = np.argmax(np.abs(u), axis=-2)
#             if u.ndim == 3:
#                 signs = np.array([np.sign(u[i, max_abs_cols[i, :], range(u.shape[-1])]) for i in range(u.shape[0])])
#             else:
#                 signs = np.sign(u[max_abs_cols, range(u.shape[-1])])
            
#             u *= signs[..., np.newaxis, :]
#             v *= signs[..., :, np.newaxis]
            
#             # Store results
#             self.left_singular_vectors = u[..., :n_components]
#             self.singular_values = s[..., :n_components]
#             self.right_singular_vectors = v[..., :n_components]
            
#     def transform(self, /, z: np.ndarray, *, direction: str) -> np.ndarray:
#         left_vectors = self.left_singular_vectors
#         if isinstance(left_vectors, torch.Tensor):
#             left_vectors = left_vectors.cpu().numpy()
            
#         right_vectors = self.right_singular_vectors
#         if isinstance(right_vectors, torch.Tensor):
#             right_vectors = right_vectors.cpu().numpy()
#         match direction:
#             case "left":
#                 return (z - self.left_mean) / self.left_std @ self.left_singular_vectors
#             case "right":
#                 return (z - self.right_mean) / self.right_std @ self.right_singular_vectors
#             case _:
#                 raise ValueError("direction must be 'left' or 'right'")
            
#     def inverse_transform(self, /, z: np.ndarray, *, direction: str) -> np.ndarray:
#         left_vectors = self.left_singular_vectors
#         if isinstance(left_vectors, torch.Tensor):
#             left_vectors = left_vectors.cpu().numpy()
            
#         right_vectors = self.right_singular_vectors
#         if isinstance(right_vectors, torch.Tensor):
#             right_vectors = right_vectors.cpu().numpy()
#         match direction:
#             case "left":
#                 return z @ self.left_singular_vectors.T
#             case "right":
#                 return z @ self.right_singular_vectors.T
#             case _:
#                 raise ValueError("direction must be 'left' or 'right'")  