import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import os
from abc import ABC, abstractmethod
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
from src.utils._config import *
from src.utils._io import *
from src.utils.plot import *
from src.utils.binning import *
from src.spectra._definition import *
from src.idm._definition import *
import argparse
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

MIN_LENGTH = 1430
BINS = np.unique(np.geomspace(1, MIN_LENGTH - 1, N_BINS).astype(int))
BINS = BINS - 1
MOVING_WINDOWS = [(BINS[i], BINS[i+1]) for i in range(len(BINS)-1)]

HOME_DIR = Path("/data/chan21/idim-debug-spectra/")
#HOME_DIR = Path("/home/chan21/idim-debug-spectra/")
bin_strategy = BinStrategy(
            type='logarithmic',
            density=3,
            start=1,
            stop=10_000
        )

def get_min_length(h5_data_path, subjects, movies):
    """Calculate minimum length across all subject pairs and movies."""
    try:
        with h5py.File(h5_data_path, 'r') as h5f:
            min_length = np.min([
                h5f[f"data/sub-NSD103_sub-NSD104_{movie}"]["observed"].shape[0] 
                for movie in movies
            ])
        return min_length
    except Exception as e:
        print(f"Error calculating min_length: {e}")
        # Return a default value or raise exception
        raise

def compute_idm_fp(args, movie=None):
    ANALYSIS_DIR = HOME_DIR / "results" / args.dir
    config = AnalysisConfig.from_yaml(ANALYSIS_DIR / "config.yaml")
    assert config.roi_names[0] == args.roi
    assert config.metric == args.metric

    config.data_path = ANALYSIS_DIR / "eigenspectra.h5"
    cache_dir = HOME_DIR / "data" / "cache" / f"idm_{args.roi}_{args.metric}"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)

    cache = DataCache(
        cache_dir = cache_dir,
        enabled=False,
        )
    analysis = IDMAnalysis(cache, bin_strategy)
    movies = config.movies
    subjects = config.subjects
    n_subjects = len(subjects)
    h5_data_path = config.data_path
    print(h5_data_path)
    
    nuisance_matrices, _ = create_nuisance_matrices(args, config)    
    anat_matrices = nuisance_matrices.get('anatomical', {})
    if type(args.motion_type) == list:
        motion_matrices = []
        for motion in args.motion_type:
            motion_matrices.append(nuisance_matrices.get(f"motion_{motion}", {}))
    else:
        motion_matrices = nuisance_matrices.get('motion', {})

    min_length = get_min_length(h5_data_path, subjects, movies)
    indices=np.arange(min_length)
    bin_assignments = [assign_data_to_geometrically_spaced_bins(
        index + 1,
        density=3,
        start=1,
        stop=10_000) for index in indices]
    unique_bins = np.unique(bin_assignments)

    if movie is not None:
        idms = []
        for movie in config.movies:
            all_spectra = np.zeros((n_subjects, n_subjects, len(unique_bins)))
            with h5py.File(h5_data_path, 'r') as h5f:
                for i, i_subject in enumerate(subjects):
                    for j, j_subject in enumerate(subjects[i+1:], i+1):
                        spectrum = h5f[f"data/{i_subject}_{j_subject}_{movie}"]["observed"][:MIN_LENGTH]
                        for n_bin, bin_id in enumerate(unique_bins):
                            all_spectra[i, j, n_bin] = spectrum[bin_assignments == bin_id].mean()
                            all_spectra[j, i, n_bin] = all_spectra[i, j, n_bin]
                idms.append(np.moveaxis(all_spectra, -1, 0))
        return analysis, idms, anat_matrices

    even_movies = config.movies[::2]
    odd_movies = config.movies[1::2]
    even_anat = np.nanmean([anat_matrices[m] for m in even_movies], axis=0) if anat_matrices else None       
    odd_anat = np.nanmean([anat_matrices[m] for m in odd_movies], axis=0) if anat_matrices else None

    idm_series = {}
    with h5py.File(h5_data_path, 'r') as h5f:
        # Initialize array to store all spectra    
        for movie in movies:
            all_spectra = np.zeros((n_subjects, n_subjects, len(unique_bins)))
            # Load spectra for each subject pair
            for i, i_subject in enumerate(subjects):
                for j, j_subject in enumerate(subjects[i+1:], i+1):
                    spectrum = h5f[f"data/{i_subject}_{j_subject}_{movie}"]["observed"][:min_length]
                    for n_bin, bin_id in enumerate(unique_bins):
                        all_spectra[i, j, n_bin] = spectrum[bin_assignments == bin_id].mean()
                        all_spectra[j, i, n_bin] = all_spectra[i, j, n_bin]
            idm_series[movie] = all_spectra

    idm_even_series = np.nanmean([idm_series[movie] for movie in movies[::2]], axis=0)
    idm_odd_series = np.nanmean([idm_series[movie] for movie in movies[1::2]], axis=0)
    
    even_idm, odd_idm = np.moveaxis(idm_even_series, -1, 0), np.moveaxis(idm_odd_series, -1, 0)
    if args.motion:
        return analysis, even_idm, odd_idm, even_anat, odd_anat, motion_matrices, unique_bins
    else:
        return analysis, even_idm, odd_idm, even_anat, odd_anat, unique_bins

def permute_idm_series(args, n_permutations=1000):
    ANALYSIS_DIR = HOME_DIR / "results" / args.dir
    config = AnalysisConfig.from_yaml(ANALYSIS_DIR / "config.yaml")
    assert config.roi_names[0] == args.roi
    assert config.metric == args.metric

    config.data_path = ANALYSIS_DIR / "eigenspectra.h5"
    cache_dir = HOME_DIR / "data" / "cache" / f"idm_{args.roi}_{args.metric}"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)

    cache = DataCache(
        cache_dir = cache_dir,
        enabled=False,
        )
    movies = config.movies
    subjects = config.subjects
    n_subjects = len(subjects)
    h5_data_path = config.data_path

    min_length = get_min_length(h5_data_path, subjects, movies)
    indices=np.arange(min_length)
    bin_assignments = [assign_data_to_geometrically_spaced_bins(
        index + 1,
        density=3,
        start=1,
        stop=10_000) for index in indices]
    unique_bins = np.unique(bin_assignments)

    nuisance_matrices, _ = create_nuisance_matrices(args, config)    
    anat_matrices = nuisance_matrices.get('anatomical', {})
    if type(args.motion_type) == list:
        motion_matrices = []
        for motion in args.motion_type:
            motion_matrices.append(nuisance_matrices.get(f"motion_{motion}", {}))
    else:
        motion_matrices = nuisance_matrices.get('motion', {})

    bin_strategy = None
    analysis = IDMAnalysis(cache, bin_strategy)

    permuted_series = {movie: [] for movie in movies}
    if args.isc:
        file = f"{HOME_DIR}/data/cache/idm_{args.roi}_{args.metric}/{args.roi}_{args.metric}_permuted_fp_isc_{str(n_permutations)}.pkl"
    else:
        file = f"{HOME_DIR}/data/cache/idm_{args.roi}_{args.metric}/{args.roi}_{args.metric}_permuted_fp_{str(n_permutations)}.pkl"
    if os.path.exists(file):
        permuted_series = pickle.load(open(file, "rb"))
    else:
        for p in tqdm(range(n_permutations)):
            with h5py.File(h5_data_path, 'r') as h5f:
                for movie in movies:
                    all_spectra = np.zeros((n_subjects, n_subjects, len(unique_bins)))
                    perm_i = np.random.permutation(subjects)
                    perm_j = np.random.permutation(subjects)
                    for i, i_subject in enumerate(perm_i):
                        for j, j_subject in enumerate(perm_j):
                            if i_subject != j_subject:
                                try:
                                    spectrum = h5f[f"data/{i_subject}_{j_subject}_{movie}"]["observed"][:min_length]
                                except:
                                    spectrum = h5f[f"data/{j_subject}_{i_subject}_{movie}"]["observed"][:min_length]
                                for n_bin, bin_id in enumerate(unique_bins):
                                    all_spectra[i, j, n_bin] = spectrum[bin_assignments == bin_id].mean()
                                    all_spectra[j, i, n_bin] = all_spectra[i, j, n_bin]
                    permuted_series[movie].append(np.moveaxis(all_spectra, -1, 0))
        with open(file, "wb") as f:
            pickle.dump(permuted_series, f)
            
    return permuted_series, unique_bins
    

def main():
    rois = ['occipital', 'ventral', 'temporal', 'semantic']
    rois_names = {'occipital': 'OCC',
              'ventral': 'VNT TMP',
              'temporal': 'LAT TMP',
              'semantic': 'SEM'}
    
    roi_corrs = {}
    roi_corrs_isc = {}
    roi_split_corrs = {}
    roi_split_corrs_isc = {}

    roi_perm_corrs = {roi: [] for roi in rois}
    roi_perm_corrs_isc = {roi: [] for roi in rois}
    roi_perm_split_corrs = {roi: [] for roi in rois}
    roi_perm_split_corrs_isc = {roi: [] for roi in rois}
    significance_isc = {roi: [] for roi in rois}
    significance = {roi: [] for roi in rois}

    for roi in rois:
        args = argparse.Namespace(
            dir=f"cov_{roi}_functional_perm",
            roi=f"{roi}",
            metric="cov",
            motion=True,
            motion_dir="/home/chan21/_individual_dim/results/",
            motion_type=["ISC", "AnnaK"],
            isc=True,
            isc_dir=f"/data/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #f"/home/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #"/data/chan21/indiv_diff_dim/results/n=39/isc_occipital_functional/eigenspectra.h5",
            anatomical=False,
            #anatomical_dir="/path/to/anatomical_matrices"
        )

        analysis, even_idm, odd_idm, even_anat, odd_anat, motion_matrices, unique_bins = compute_idm_fp(args)
        movies = list(motion_matrices[0].keys())
        even_motion1 = np.nanmean([motion_matrices[0][m] for m in movies[::2]], axis=0)
        odd_motion1 = np.nanmean([motion_matrices[0][m] for m in movies[1::2]], axis=0)
        even_motion2 = np.nanmean([motion_matrices[1][m] for m in movies[::2]], axis=0)
        odd_motion2 = np.nanmean([motion_matrices[1][m] for m in movies[1::2]], axis=0)

    #     if args.isc:
    #         n_bins = len(even_idm)
    #         correlations = []
    #         residual1 = []
    #         residual2 = []
    #         for n_bin in range(n_bins):
    #             _, rx, ry = analysis._fully_partial_correlation(
    #                 get_upper_triangle(even_idm[n_bin]),
    #                 get_upper_triangle(odd_idm[n_bin]),
    #                 get_upper_triangle(even_motion1),
    #                 get_upper_triangle(odd_motion1))
    #             _, rx, ry = analysis._fully_partial_correlation(
    #                 rx,
    #                 ry,
    #                 get_upper_triangle(even_motion2),
    #                 get_upper_triangle(odd_motion2))
    #             corr, res1, res2 = analysis._fully_partial_correlation(
    #                 rx,
    #                 ry,
    #                 get_upper_triangle(even_anat),
    #                 get_upper_triangle(odd_anat))
    #             correlations.append(corr)
    #             residual1.append(res1)
    #             residual2.append(res2)
    #         roi_corrs_isc[roi] = correlations
    #         print(roi, correlations)
    #         diagonals = np.zeros((len(correlations), len(correlations)))
    #         for i, corr in enumerate(correlations):
    #             residual1_tri = residual1[i]
    #             for j, c in enumerate(correlations):
    #                 diagonals[i, j] = np.corrcoef(residual1_tri, residual2[j])[0, 1]
    #         roi_split_corrs_isc[roi] = diagonals
    #     else:
    #         n_bins = len(even_idm)
    #         correlations = []
    #         residual1 = []
    #         residual2 = []
    #         for n_bin in range(n_bins):
    #             # # Visualization
    #             # idm_e = - zscore_matrix(even_idm[n_bin])
    #             # idm_o = - zscore_matrix(odd_idm[n_bin])
    #             # # Set diagonal to 0
    #             # np.fill_diagonal(idm_e, np.min(idm_e))
    #             # np.fill_diagonal(idm_o, np.min(idm_o))
    #             # save_matrix(idm_e, filename=f"idm_even_diff_{n_bin}.svg", style='coolwarm')
    #             # save_matrix(idm_o, filename=f"idm_odd_diff_{n_bin}.svg", style='coolwarm')
    #             # if n_bin == 3:
    #             #     break
    #             _, rx, ry = analysis._fully_partial_correlation(
    #                 get_upper_triangle(even_idm[n_bin]),
    #                 get_upper_triangle(odd_idm[n_bin]),
    #                 get_upper_triangle(even_motion1),
    #                 get_upper_triangle(odd_motion1))
    #             corr, rx, ry = analysis._fully_partial_correlation(
    #                 rx,
    #                 ry,
    #                 get_upper_triangle(even_motion2),
    #                 get_upper_triangle(odd_motion2))
    #             correlations.append(corr)
    #             residual1.append(rx)
    #             residual2.append(ry)
    #         roi_corrs[roi] = correlations
    #         print(roi, correlations)
    #         diagonals = np.zeros((len(correlations), len(correlations)))
    #         for i, corr in enumerate(correlations):
    #             residual1_tri = residual1[i]
    #             for j, c in enumerate(correlations):
    #                 diagonals[i, j] = np.corrcoef(residual1_tri, residual2[j])[0, 1]
    #         roi_split_corrs[roi] = diagonals

    # for roi in rois:
    #     args = argparse.Namespace(
    #         dir=f"cov_{roi}_functional_perm",
    #         roi=f"{roi}",
    #         metric="cov",
    #         motion=True,
    #         motion_dir="/home/chan21/_individual_dim/results/",
    #         motion_type=["ISC", "AnnaK"],
    #         isc=False,
    #         isc_dir=f"/home/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #"/data/chan21/indiv_diff_dim/results/n=39/isc_occipital_functional/eigenspectra.h5",
    #         anatomical=False,
    #         #anatomical_dir="/path/to/anatomical_matrices"
    #     )

    #     ANALYSIS_DIR = HOME_DIR / "results" / args.dir
    #     config = AnalysisConfig.from_yaml(ANALYSIS_DIR / "config.yaml")
    #     nuisance_matrices, _ = create_nuisance_matrices(args, config)    
    #     anat_matrices = nuisance_matrices.get('anatomical', {})
    #     even_movies = config.movies[::2]
    #     odd_movies = config.movies[1::2]
    #     even_anat = np.nanmean([anat_matrices[m] for m in even_movies], axis=0) if anat_matrices else None
    #     odd_anat = np.nanmean([anat_matrices[m] for m in odd_movies], axis=0) if anat_matrices else None       

    #     permuted_series, unique_bins = permute_idm_series(args)

    #     for perm_idx in range(1000):
    #         permuted_idm1 = np.nanmean([permuted_series[movie][perm_idx] for movie in config.movies[::2]], axis=0)
    #         permuted_idm2 = np.nanmean([permuted_series[movie][perm_idx] for movie in config.movies[1::2]], axis=0)

    #         if args.isc:
    #             corr = np.zeros((len(unique_bins)))
    #             for n_bin in range(len(unique_bins)):
    #                 _, rx, ry = analysis._fully_partial_correlation(
    #                     get_upper_triangle(permuted_idm1[n_bin]),
    #                     get_upper_triangle(permuted_idm2[n_bin]),
    #                     get_upper_triangle(even_motion1),
    #                     get_upper_triangle(odd_motion1))
    #                 _, rx, ry = analysis._fully_partial_correlation(
    #                     rx,
    #                     ry,
    #                     get_upper_triangle(even_motion2),
    #                     get_upper_triangle(odd_motion2))
    #                 r, _, _ = analysis._fully_partial_correlation(
    #                     rx,
    #                     ry,
    #                     get_upper_triangle(even_anat),
    #                     get_upper_triangle(odd_anat))
    #                 corr[n_bin] = r
    #             roi_perm_corrs_isc[roi].append(corr)
    #         else:
    #             corr = np.zeros((n_bins))
    #             for n_bin in range(n_bins):
    #                 _, rx, ry = analysis._fully_partial_correlation(
    #                     get_upper_triangle(permuted_idm1[n_bin]),
    #                     get_upper_triangle(permuted_idm2[n_bin]),
    #                     get_upper_triangle(even_motion1),
    #                     get_upper_triangle(odd_motion1))
    #                 r, rx, ry = analysis._fully_partial_correlation(
    #                     rx,
    #                     ry,
    #                     get_upper_triangle(even_motion2),
    #                     get_upper_triangle(odd_motion2))
    #                 #print(r)
    #                 corr[n_bin] = r
    #             roi_perm_corrs[roi].append(corr)

    #         diagonals = np.zeros((len(unique_bins), len(unique_bins)))
    #         for i, bin in enumerate(unique_bins):
    #             permuted_idm1_tri = permuted_idm1[i][np.triu_indices(permuted_idm1[i].shape[1], k=1)]
    #             for j, bin in enumerate(unique_bins):
    #                 diagonals[i, j] = np.corrcoef(permuted_idm1_tri, permuted_idm2[j][np.triu_indices(permuted_idm2[j].shape[1], k=1)])[0, 1]
    #         if args.isc:
    #             roi_perm_split_corrs_isc[roi].append(diagonals)
    #         else:
    #             roi_perm_split_corrs[roi].append(diagonals)


    for roi in rois:
        args = argparse.Namespace(
            dir=f"cov_{roi}_functional_perm",
            roi=f"{roi}",
            metric="cov",
            motion=True,
            motion_dir="/home/chan21/_individual_dim/results/",
            motion_type=["ISC", "AnnaK"],
            isc=True,
            isc_dir=f"/home/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #"/data/chan21/indiv_diff_dim/results/n=39/isc_occipital_functional/eigenspectra.h5",
            anatomical=False,
            #anatomical_dir="/path/to/anatomical_matrices"
        )

        ANALYSIS_DIR = HOME_DIR / "results" / args.dir
        config = AnalysisConfig.from_yaml(ANALYSIS_DIR / "config.yaml")
        nuisance_matrices, _ = create_nuisance_matrices(args, config)    
        anat_matrices = nuisance_matrices.get('anatomical', {})
        even_movies = config.movies[::2]
        odd_movies = config.movies[1::2]
        even_anat = np.nanmean([anat_matrices[m] for m in even_movies], axis=0) if anat_matrices else None
        odd_anat = np.nanmean([anat_matrices[m] for m in odd_movies], axis=0) if anat_matrices else None       

        permuted_series, unique_bins = permute_idm_series(args)

        for perm_idx in range(1000):
            permuted_idm1 = np.nanmean([permuted_series[movie][perm_idx] for movie in config.movies[::2]], axis=0)
            permuted_idm2 = np.nanmean([permuted_series[movie][perm_idx] for movie in config.movies[1::2]], axis=0)

            if args.isc:
                corr = np.zeros((len(unique_bins)))
                for n_bin in range(len(unique_bins)):
                    _, rx, ry = analysis._fully_partial_correlation(
                        get_upper_triangle(permuted_idm1[n_bin]),
                        get_upper_triangle(permuted_idm2[n_bin]),
                        get_upper_triangle(even_motion1),
                        get_upper_triangle(odd_motion1))
                    _, rx, ry = analysis._fully_partial_correlation(
                        rx,
                        ry,
                        get_upper_triangle(even_motion2),
                        get_upper_triangle(odd_motion2))
                    r, _, _ = analysis._fully_partial_correlation(
                        rx,
                        ry,
                        get_upper_triangle(even_anat),
                        get_upper_triangle(odd_anat))
                    corr[n_bin] = r
                roi_perm_corrs_isc[roi].append(corr)
            else:
                corr = np.zeros((n_bins))
                for n_bin in range(n_bins):
                    _, rx, ry = analysis._fully_partial_correlation(
                        get_upper_triangle(permuted_idm1[n_bin]),
                        get_upper_triangle(permuted_idm2[n_bin]),
                        get_upper_triangle(even_motion1),
                        get_upper_triangle(odd_motion1))
                    r, rx, ry = analysis._fully_partial_correlation(
                        rx,
                        ry,
                        get_upper_triangle(even_motion2),
                        get_upper_triangle(odd_motion2))
                    #print(r)
                    corr[n_bin] = r
                roi_perm_corrs[roi].append(corr)

            diagonals = np.zeros((len(unique_bins), len(unique_bins)))
            for i, bin in enumerate(unique_bins):
                permuted_idm1_tri = permuted_idm1[i][np.triu_indices(permuted_idm1[i].shape[1], k=1)]
                for j, bin in enumerate(unique_bins):
                    diagonals[i, j] = np.corrcoef(permuted_idm1_tri, permuted_idm2[j][np.triu_indices(permuted_idm2[j].shape[1], k=1)])[0, 1]
            if args.isc:
                roi_perm_split_corrs_isc[roi].append(diagonals)
            else:
                roi_perm_split_corrs[roi].append(diagonals)

    alpha = 0.05
        
    p_values = {roi: [] for roi in rois}
    for roi in rois:
        corrs = roi_corrs[roi]
        perm_corrs = roi_perm_corrs[roi]
        for n_bin, bin in enumerate(unique_bins):
            observed = corrs[n_bin]
            permuted = np.array([corr[n_bin] for corr in perm_corrs])
            n_permutations = len(permuted)
            n_exceeding = np.sum(permuted >= observed, axis=0)
            effect_size = (observed - np.mean(permuted)) / np.std(permuted)
            pvalue = n_exceeding / n_permutations
            corrected_pvalue = min(pvalue * len(unique_bins), 1)
            p_values[roi].append(corrected_pvalue)
            print(f"{roi}: {bin} {round(observed, 3)}; {round(effect_size, 3)}; {corrected_pvalue}")

    p_values_isc = {roi: [] for roi in rois}
    for roi in rois:
        corrs = roi_corrs_isc[roi]
        perm_corrs = roi_perm_corrs_isc[roi]
        for n_bin, bin in enumerate(unique_bins):
            observed = corrs[n_bin]
            permuted = np.array([corr[n_bin] for corr in perm_corrs])
            n_permutations = len(permuted)
            n_exceeding = np.sum(permuted >= observed, axis=0)
            effect_size = (observed - np.mean(permuted)) / np.std(permuted)
            pvalue = n_exceeding / n_permutations
            corrected_pvalue = min(pvalue * len(unique_bins), 1)
            p_values_isc[roi].append(corrected_pvalue)
            print(f"{roi}: {bin} {round(observed, 3)}; {round(effect_size, 3)}; {corrected_pvalue}")

if __name__ == "__main__":
    main()