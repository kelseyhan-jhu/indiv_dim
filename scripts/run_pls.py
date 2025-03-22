#!/usr/bin/env python3

import sys
PROJECT_ROOT = "/scratch4/mbonner5/chan21/indiv_dim"
sys.path.append(PROJECT_ROOT)

from src.utils._pls import PLSSVD, setup_logging, load_subject_movie_data, get_common_indices, extract_data

print(PROJECT_ROOT)

import os
import argparse
import numpy as np
import h5py
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import combinations
import yaml
from sklearn.decomposition import PCA

PERM_BATCH_SIZE = 20  # For permutation processing

def parse_args():
    parser = argparse.ArgumentParser(description='PLS Analysis Pipeline')
    parser.add_argument('--roi', type=str, required=True)
    parser.add_argument('--batch-start', type=int, required=True)
    parser.add_argument('--batch-end', type=int, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--perform-permutations', type=str, required=True)
    parser.add_argument('--n-permutations', type=int, required=True)
    parser.add_argument('--perform-downsampling', type=str, required=True)
    parser.add_argument('--n-downsample', type=int, required=True)
    parser.add_argument('--alignment', type=str, required=True)
    parser.add_argument('--random-state', type=int, required=True)
    parser.add_argument('--block-size', type=int, required=True)
    parser.add_argument('--subjects', type=str, required=True)
    parser.add_argument('--movies', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    
    args = parser.parse_args()
    
    # Convert string lists to actual lists
    args.subjects = args.subjects.split(',')
    args.movies = args.movies.split(',')
    return args

def save_config(args, output_path):
    """
    Save configuration to a YAML file with specific formatting.
    
    Args:
        args: Namespace object containing configuration
        output_path: Path where to save the YAML file
    """
    config = {
        'subjects': args.subjects if isinstance(args.subjects, list) else args.subjects.split(','),
        'movies': args.movies if isinstance(args.movies, list) else args.movies.split(','),
        'metric': args.metric,
        'roi_names': args.roi if isinstance(args.roi, list) else args.roi.split(','),
        'alignment': args.alignment,
        'perform_permutations': True if args.perform_permutations == "True" else False,
        'n_permutations': args.n_permutations,
        'perform_downsampling': True if args.perform_permutations == "True" else False,
        'n_downsample': args.n_downsample,
        'random_state': args.random_state,
        'block_size': args.block_size
    }
    
    # Save with proper YAML formatting
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def get_subject_pairs(subjects, batch_start, batch_end):
    """Generate subject pairs for the specific batch range"""
    all_pairs = list(combinations(subjects, 2))
    total_pairs = len(all_pairs)
    
    # Ensure indices are within bounds
    start_idx = max(0, min(batch_start, total_pairs - 1))
    end_idx = max(start_idx, min(batch_end, total_pairs - 1))
    
    logging.info(f"Total pairs: {total_pairs}")
    logging.info(f"Processing indices {start_idx} to {end_idx}")
    
    return all_pairs[start_idx:end_idx + 1]

def validate_subject_data(sub_xarr):
    """Validate that all subject data is loaded properly"""
    try:
        for subject, movie_data in sub_xarr.items():
            for movie, data in movie_data.items():
                if data is None or not isinstance(data, np.ndarray):
                    logging.warning(f"Missing or invalid data for subject {subject}, movie {movie}")
                    return False
                if data.size == 0:
                    logging.warning(f"Empty data array for subject {subject}, movie {movie}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Error validating data: {str(e)}")
        return False

def compute_permutation_batch(args):
    """Process a batch of permutations with reproducible permutations"""
    pls, x_test, y_test, metric, block_timepoints, batch_size, batch_start, random_state = args
    try:
        batch_spectra = []
        
        for perm_idx in range(batch_start, batch_start + batch_size):
            # Create two different but deterministic seeds for x and y
            seed_x = random_state + perm_idx * 2  # Even numbers for x
            seed_y = random_state + perm_idx * 2 + 1  # Odd numbers for y
            
            # Create separate RNGs for x and y
            rng_x = np.random.default_rng(seed_x)
            rng_y = np.random.default_rng(seed_y)
            
            # Generate permutations
            perm_x = np.concatenate([block_timepoints[i] for i in rng_x.permutation(len(block_timepoints))])
            perm_y = np.concatenate([block_timepoints[i] for i in rng_y.permutation(len(block_timepoints))])
            
            x_transformed = pls.transform(x_test[perm_x], direction="left")
            y_transformed = pls.transform(y_test[perm_y], direction="right")
            
            min_components = min(x_transformed.shape[1], y_transformed.shape[1])
            if metric == 'corr':
                spectrum = [np.corrcoef(x_transformed[:, k], y_transformed[:, k])[0, 1] 
                           for k in range(min_components)]
            else:  # 'cov'
                spectrum = np.diag(np.cov(x_transformed, y_transformed, rowvar=False)[:min_components, min_components:])
                
            if perm_idx % PERM_BATCH_SIZE == 0:
                logging.info(f"Processed permutation {perm_idx}")    
            batch_spectra.append(spectrum)
        
        return np.array(batch_spectra)
    except Exception as e:
        logging.error(f"Error in permutation batch: {str(e)}")
        return None

def run_permutation_tests(pls, x_test, y_test, metric, n_permutations, block_size=1, random_state=42):
    """Run permutation tests with batching and reproducible permutations"""
    try:
        ntimepoints = x_test.shape[0]
        nchunk = ntimepoints // block_size
        chunk_sizes = [block_size] * nchunk + ([ntimepoints % block_size] if ntimepoints % block_size else [])
        block_indices = np.cumsum([0] + chunk_sizes[:-1])
        block_timepoints = [np.arange(start, start + size) 
                          for start, size in zip(block_indices, chunk_sizes)]

        # Process permutations in batches
        n_batches = (n_permutations + PERM_BATCH_SIZE - 1) // PERM_BATCH_SIZE
        batch_sizes = [PERM_BATCH_SIZE] * (n_batches - 1) + [n_permutations % PERM_BATCH_SIZE or PERM_BATCH_SIZE]
        
        # Create args list with batch start indices for permutation numbering
        args_list = []
        current_perm = 0
        for batch_size in batch_sizes:
            args_list.append((pls, x_test, y_test, metric, block_timepoints, 
                            batch_size, current_perm, random_state))
            current_perm += batch_size
        
        # Use ProcessPoolExecutor for parallel processing
        n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(compute_permutation_batch, args_list))
            
        if any(result is None for result in results):
            raise ValueError("One or more permutation batches failed")
            
        return np.concatenate(results)
    except Exception as e:
        logging.error(f"Error in permutation tests: {str(e)}")
        return None

def process_subject_pair(sub_pair, args, output_base):
    """Process a single subject pair"""
    try:
        output_filepath = f"{output_base}/eigenspectra_{sub_pair[0]}_{sub_pair[1]}.h5"
        
        # Skip if output already exists
        if os.path.exists(output_filepath):
            logging.info(f"Output already exists for {sub_pair}. Skipping.")
            return True
            
        logging.info(f"Processing subject pair: {sub_pair[0]} & {sub_pair[1]}")

        # Load and validate data
        sub_xarr = {subj: {movie: load_subject_movie_data(subj, movie, args.roi, args.data_path)
                          for movie in args.movies}
                   for subj in sub_pair}

        if not validate_subject_data(sub_xarr):
            return False

        nonzeros_common = get_common_indices(sub_xarr, sub_pair, args.movies, args.n_downsample)
        if nonzeros_common.size == 0:
            logging.warning(f"No common non-zero voxels for {sub_pair}. Skipping.")
            return False

        sub_xarr = extract_data(sub_xarr, sub_pair, args.movies, nonzeros_common)

        # Process each movie
        with h5py.File(output_filepath, 'w') as h5f:
            h5f.attrs.update({
                'subjects': ','.join(sub_pair),
                'movies': ','.join(args.movies),
                'roi_name': args.roi,
                'metric': args.metric,
                'perform_permutations': args.perform_permutations,
                'perform_downsampling': args.perform_downsampling,
                'n_permutations': args.n_permutations if args.perform_permutations else 0,
                'n_downsample': args.n_downsample if args.perform_downsampling else 0
            })

            for test_movie in args.movies:
                training_movies = [m for m in args.movies if m != test_movie]
                
                if args.metric == 'isc':
                    x_data = sub_xarr[sub_pair[0]][test_movie].T
                    y_data = sub_xarr[sub_pair[1]][test_movie].T
                    spectrum = np.array([np.corrcoef(x_data[:, v], y_data[:, v])[0, 1] 
                                      for v in range(nonzeros_common.shape[0])])
                    print(spectrum.shape)
                    print(np.mean(spectrum))
                    permutation_spectra = None

                elif args.metric == 'recon':
                    x_train = np.concatenate([sub_xarr[sub_pair[0]][m].T for m in training_movies])
                    y_train = np.concatenate([sub_xarr[sub_pair[1]][m].T for m in training_movies])
                    x_test = sub_xarr[sub_pair[0]][test_movie].T
                    y_test = sub_xarr[sub_pair[1]][test_movie].T

                    # Perform PLS analysis
                    pls = PLSSVD()
                    pls.fit(x_train, y_train, random_state=args.random_state)
                    x_transformed = pls.transform(x_test, direction='left')
                    y_transformed = pls.transform(y_test, direction='right')

                    x_reconstructed = pls.inverse_transform(x_transformed, direction='left')
                    y_reconstructed = pls.inverse_transform(y_transformed, direction='right')

                    # log dimensions of transformed and reconstructed data
                    # logging.info(f"Dimensions of transformed data: {x_transformed.shape}")
                    # logging.info(f"Dimesnions of reconstructed data: {x_reconstructed.shape}")
                    # #log correlation between original and reconstructed data
                    # logging.info(f"Correlation between original and reconstructed data: {np.corrcoef(x_test.flatten(), x_reconstructed.flatten())[0, 1]}")
                    # logging.info(f"Correlation between original and reconstructed data: {np.corrcoef(y_test.flatten(), y_reconstructed.flatten())[0, 1]}")

                    spectrum = np.array([np.corrcoef(x_reconstructed[:, v], y_reconstructed[:, v])[0, 1] 
                                      for v in range(nonzeros_common.shape[0])])
                    # log spectrum
                    logging.info(f"Spectrum: {spectrum}")
                    permutation_spectra = None

                elif args.metric == 'pca':
                    x_data = sub_xarr[sub_pair[0]][test_movie].T
                    y_data = sub_xarr[sub_pair[1]][test_movie].T
                    pca_x = PCA(n_components=10)
                    pca_y = PCA(n_components=10)
                    ## log the variance explained by each component
                    #var_x = pca_x.fit(x_data).explained_variance_ratio_
                    #var_y = pca_y.fit(y_data).explained_variance_ratio_
                    #logging.info(f"Variance explained by each component: {var_x}")
                    #logging.info(f"Variance explained by each component: {var_y}")
                    x_transformed = pca_x.fit_transform(x_data) # dimensions = timepoints x components
                    y_transformed = pca_y.fit_transform(y_data)
                    x_reconstructed = pca_x.inverse_transform(x_transformed)
                    y_reconstructed = pca_y.inverse_transform(y_transformed)
                    # log dimensions of transformed and reconstructed data
                    #logging.info(f"Dimensions of transformed data: {x_transformed.shape}")
                    #logging.info(f"Dimesnions of reconstructed data: {x_reconstructed.shape}")
                    # log correlation between original and reconstructed data
                    #logging.info(f"Correlation between original and reconstructed data: {np.corrcoef(x_data.flatten(), x_reconstructed.flatten())[0, 1]}")
                    #logging.info(f"Correlation between original and reconstructed data: {np.corrcoef(y_data.flatten(), y_reconstructed.flatten())[0, 1]}")

                    spectrum = np.array([np.corrcoef(x_reconstructed[:, v], y_reconstructed[:, v])[0, 1] 
                                      for v in range(nonzeros_common.shape[0])])
                    # log spectrum
                    #logging.info(f"Spectrum: {spectrum}")
                    permutation_spectra = None
                else:
                    # Prepare data for PLS
                    x_train = np.concatenate([sub_xarr[sub_pair[0]][m].T for m in training_movies])
                    y_train = np.concatenate([sub_xarr[sub_pair[1]][m].T for m in training_movies])
                    x_test = sub_xarr[sub_pair[0]][test_movie].T
                    y_test = sub_xarr[sub_pair[1]][test_movie].T
                    #logging.info(f"Data shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")

                    # Perform PLS analysis
                    pls = PLSSVD()
                    #logging.info(f"Fitting PLS model for {sub_pair} and {test_movie}")
                    pls.fit(x_train, y_train, random_state=args.random_state)
                    #logging.info(f"Transforming data for {sub_pair} and {test_movie}")
                    x_transformed = pls.transform(x_test, direction='left')
                    y_transformed = pls.transform(y_test, direction='right')

                    if args.alignment == "functional":
                        # Compute observed spectrum
                        min_components = min(x_transformed.shape[1], y_transformed.shape[1])
                        if args.metric == 'corr':
                            logging.info(f"Computing correlations for {sub_pair} and {test_movie}")
                            spectrum = [np.corrcoef(x_transformed[:, k], y_transformed[:, k])[0, 1] 
                                      for k in range(min_components)]
                        else:  # 'cov'
                            logging.info(f"Computing covariances for {sub_pair} and {test_movie}")
                            spectrum = np.diag(np.cov(x_transformed, y_transformed, rowvar=False)
                                            [:min_components, min_components:])
                            #logging.info(f"Spectra: {spectrum}")
                        # Perform permutations if enabled
                        if args.perform_permutations == 'True':
                            permutation_spectra = run_permutation_tests(
                                pls, x_test, y_test, args.metric, args.n_permutations,
                                block_size=args.block_size
                            )
                            if permutation_spectra is None:
                                raise ValueError("Permutation testing failed")
                        else:
                            permutation_spectra = None

                    else:  # anatomical alignment
                        x_recon_x = pls.recon(x_transformed, direction="left")
                        y_recon_x = pls.recon(y_transformed, direction="left")
                        x_recon_y = pls.recon(x_transformed, direction="right")
                        y_recon_y = pls.recon(y_transformed, direction="right")

                        spectrum_x = [np.corrcoef(x_recon_x[:, i], y_recon_x[:, i])[0, 1] 
                                    for i in range(x_recon_x.shape[1])]
                        spectrum_y = [np.corrcoef(x_recon_y[:, i], y_recon_y[:, i])[0, 1] 
                                    for i in range(x_recon_y.shape[1])]
                        
                        spectrum = np.mean([spectrum_x, spectrum_y], axis=0)
                        permutation_spectra = None
                        logging.info(f"Computed anatomical alignment spectrum for {sub_pair} and {test_movie}")

                # Save results
                group_name = f"data/{sub_pair[0]}_{sub_pair[1]}_{test_movie}"
                group = h5f.create_group(group_name)
                group.create_dataset('observed', data=np.array(spectrum))
                if permutation_spectra is not None:
                    group.create_dataset('permutations', data=permutation_spectra)

        logging.info(f"Successfully processed pair {sub_pair}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing pair {sub_pair}: {str(e)}")
        if os.path.exists(output_filepath):
            os.remove(output_filepath)  # Clean up partial output file
        return False


def main():
    args = parse_args()
    setup_logging()
    
    try:
        logging.info(f"Processing ROI: {args.roi}")
        logging.info(f"Processing batch range: {args.batch_start}-{args.batch_end}")
        
        # Generate output path
        output_base = f"./results/{args.metric}_{args.roi}_{args.alignment}"
        if args.perform_permutations == "True":
            output_base += '_perm' 
        if args.perform_downsampling == "True":
            output_base += f'_downsample{args.n_downsample}'
        os.makedirs(output_base, exist_ok=True)

        # Save config to file
        save_config(args, f"{output_base}/config.yaml")
        
        # Get subject pairs for this batch
        subject_pairs = get_subject_pairs(args.subjects, args.batch_start, args.batch_end)
        if not subject_pairs:
            logging.error("No valid pairs to process")
            sys.exit(1)
            
        logging.info(f"Processing {len(subject_pairs)} subject pairs")
        
        success_count = 0
        failed_pairs = []
        
        # Process each pair
        for sub_pair in subject_pairs:
            if process_subject_pair(sub_pair, args, output_base):
                success_count += 1
            else:
                failed_pairs.append(sub_pair)
        
        # Log summary
        logging.info(f"Batch processing complete. Successfully processed {success_count}/{len(subject_pairs)} pairs")
        if failed_pairs:
            logging.warning(f"Failed pairs: {failed_pairs}")
            sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()