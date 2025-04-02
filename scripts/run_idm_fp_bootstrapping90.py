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

    min_length = np.min([h5f[f"data/{i_subject}_{j_subject}_{movie}"]["observed"].shape[0] for i_subject in subjects for j_subject in subjects for movie in movies])
    bin_assignments = bin_strategy.get_bins(length=min_length)
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
        return analysis, even_idm, odd_idm, even_anat, odd_anat, motion_matrices
    else:
        return analysis, even_idm, odd_idm, even_anat, odd_anat


def bootstrap_idm_series(analysis, args, n_bootstraps=1000):
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

    ANALYSIS_DIR = HOME_DIR / "results" / args.dir
    config = AnalysisConfig.from_yaml(ANALYSIS_DIR / "config.yaml")
    nuisance_matrices, _ = create_nuisance_matrices(args, config)    
    anat_matrices = nuisance_matrices.get('anatomical', {})
    even_movies = config.movies[::2]
    odd_movies = config.movies[1::2]
    even_anat = np.nanmean([anat_matrices[m] for m in even_movies], axis=0) if anat_matrices else None       
    odd_anat = np.nanmean([anat_matrices[m] for m in odd_movies], axis=0) if anat_matrices else None
    if type(args.motion_type) == list:
        motion_matrices = []
        for motion in args.motion_type:
            motion_matrices.append(nuisance_matrices.get(f"motion_{motion}", {}))
        even_motion1 = np.nanmean([motion_matrices[0][m] for m in movies[::2]], axis=0)
        odd_motion1 = np.nanmean([motion_matrices[0][m] for m in movies[1::2]], axis=0)
        even_motion2 = np.nanmean([motion_matrices[1][m] for m in movies[::2]], axis=0)
        odd_motion2 = np.nanmean([motion_matrices[1][m] for m in movies[1::2]], axis=0)
    else:
        motion_matrices = nuisance_matrices.get('motion', {})
        even_motion1 = np.nanmean([motion_matrices[m] for m in movies[::2]], axis=0)
        odd_motion1 = np.nanmean([motion_matrices[m] for m in movies[1::2]], axis=0)
    
            
    n_bins = len(unique_bins)

    bootstrapped_correlations = [] 
    for boot_idx in tqdm(range(n_bootstraps)):
        if args.isc:
            file = f"{HOME_DIR}/data/cache/idm_{args.roi}_{args.metric}/{args.roi}_{args.metric}_bootstrapped90_fp_isc_{str(boot_idx)}.pkl"
        else:
            file = f"{HOME_DIR}/data/cache/idm_{args.roi}_{args.metric}/{args.roi}_{args.metric}_bootstrapped90_{str(boot_idx)}.pkl"
        if os.path.exists(file):
            with open(file, "rb") as f:
                bootstrapped_series, subject_idx = pickle.load(f)
            boot_idm1 = np.nanmean([bootstrapped_series[movie] for movie in even_movies], axis=0)
            boot_idm2 = np.nanmean([bootstrapped_series[movie] for movie in config.movies[1::2]], axis=0)
            boot_idm1 = np.moveaxis(boot_idm1, -1, 0)
            boot_idm2 = np.moveaxis(boot_idm2, -1, 0)
            bootstrapped_even_motion1 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
            bootstrapped_odd_motion1 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
            bootstrapped_even_motion2 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
            bootstrapped_odd_motion2 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
            for i, i_subject in enumerate([subjects[idx] for idx in subject_idx]):
                for j, j_subject in enumerate([subjects[idx] for idx in subject_idx]):
                    if config.subjects.index(i_subject) < config.subjects.index(j_subject):
                        bootstrapped_even_motion1[i, j] = even_motion1[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                        bootstrapped_even_motion1[j, i] = bootstrapped_even_motion1[i, j]
                        bootstrapped_odd_motion1[i, j] = odd_motion1[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                        bootstrapped_odd_motion1[j, i] = bootstrapped_odd_motion1[i, j]
                        bootstrapped_even_motion2[i, j] = even_motion2[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                        bootstrapped_even_motion2[j, i] = bootstrapped_even_motion2[i, j]
                        bootstrapped_odd_motion2[i, j] = odd_motion2[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                        bootstrapped_odd_motion2[j, i] = bootstrapped_odd_motion2[i, j]
                    elif config.subjects.index(i_subject) > config.subjects.index(j_subject):
                        bootstrapped_even_motion1[i, j] = even_motion1[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                        bootstrapped_even_motion1[j, i] = bootstrapped_even_motion1[i, j]
                        bootstrapped_odd_motion1[i, j] = odd_motion1[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                        bootstrapped_odd_motion1[j, i] = bootstrapped_odd_motion1[i, j]
                        bootstrapped_even_motion2[i, j] = even_motion2[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                        bootstrapped_even_motion2[j, i] = bootstrapped_even_motion2[i, j]
                        bootstrapped_odd_motion2[i, j] = odd_motion2[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                        bootstrapped_odd_motion2[j, i] = bootstrapped_odd_motion2[i, j]
            if args.isc:
                bootstrapped_even_anat = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
                bootstrapped_odd_anat = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
                for i, i_subject in enumerate([subjects[idx] for idx in subject_idx]):
                    for j, j_subject in enumerate([subjects[idx] for idx in subject_idx]):
                        if config.subjects.index(i_subject) < config.subjects.index(j_subject):
                            bootstrapped_even_anat[i, j] = even_anat[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                            bootstrapped_even_anat[j, i] = bootstrapped_even_anat[i, j]
                            bootstrapped_odd_anat[i, j] = odd_anat[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                            bootstrapped_odd_anat[j, i] = bootstrapped_odd_anat[i, j]
                        elif config.subjects.index(i_subject) > config.subjects.index(j_subject):
                            bootstrapped_even_anat[i, j] = even_anat[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                            bootstrapped_even_anat[j, i] = bootstrapped_even_anat[i, j]
                            bootstrapped_odd_anat[i, j] = odd_anat[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                            bootstrapped_odd_anat[j, i] = bootstrapped_odd_anat[i, j]
                corr = np.zeros((n_bins))
                for n_bin in range(n_bins):
                    _, rx, ry = analysis._fully_partial_correlation(
                        get_filtered_upper_triangle(boot_idm1[n_bin], subject_idx),
                        get_filtered_upper_triangle(boot_idm2[n_bin], subject_idx),
                        get_filtered_upper_triangle(bootstrapped_even_motion1, subject_idx),
                        get_filtered_upper_triangle(bootstrapped_odd_motion1, subject_idx))
                    _, rx, ry = analysis._fully_partial_correlation(
                        rx,
                        ry,
                        get_filtered_upper_triangle(bootstrapped_even_motion2, subject_idx),
                        get_filtered_upper_triangle(bootstrapped_odd_motion2, subject_idx))
                    r, _, _ = analysis._fully_partial_correlation(
                        rx,
                        ry,
                        get_filtered_upper_triangle(bootstrapped_even_anat, subject_idx),
                        get_filtered_upper_triangle(bootstrapped_odd_anat, subject_idx))
                    corr[n_bin] = r
            else: 
                corr = np.zeros((n_bins))
                for n_bin in range(n_bins):
                    _, rx, ry = analysis._fully_partial_correlation(
                        get_filtered_upper_triangle(boot_idm1[n_bin], subject_idx),
                        get_filtered_upper_triangle(boot_idm2[n_bin], subject_idx),
                        get_filtered_upper_triangle(bootstrapped_even_motion1, subject_idx),
                        get_filtered_upper_triangle(bootstrapped_odd_motion1, subject_idx))
                    r, rx, ry = analysis._fully_partial_correlation(
                        rx,
                        ry,
                        get_filtered_upper_triangle(bootstrapped_even_motion2, subject_idx),
                        get_filtered_upper_triangle(bootstrapped_odd_motion2, subject_idx))
                    corr[n_bin] = r
            bootstrapped_correlations.append(corr)
            continue
        
        bootstrapped_series = {}
        subject_idx = np.random.choice(n_subjects, size=int(n_subjects*0.9), replace=True)
        if anat_matrices:
            bootstrapped_even_anat = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
            bootstrapped_odd_anat = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
            for i, i_subject in enumerate([subjects[idx] for idx in subject_idx]):
                for j, j_subject in enumerate([subjects[idx] for idx in subject_idx]):
                    if config.subjects.index(i_subject) < config.subjects.index(j_subject):
                        bootstrapped_even_anat[i, j] = even_anat[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                        bootstrapped_even_anat[j, i] = bootstrapped_even_anat[i, j]
                        bootstrapped_odd_anat[i, j] = odd_anat[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                        bootstrapped_odd_anat[j, i] = bootstrapped_odd_anat[i, j]
                    elif config.subjects.index(i_subject) > config.subjects.index(j_subject):
                        bootstrapped_even_anat[i, j] = even_anat[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                        bootstrapped_even_anat[j, i] = bootstrapped_even_anat[i, j]
                        bootstrapped_odd_anat[i, j] = odd_anat[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                        bootstrapped_odd_anat[j, i] = bootstrapped_odd_anat[i, j]
        with h5py.File(h5_data_path, 'r') as h5f:
            for movie in movies:
                all_spectra = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9), len(unique_bins)))
                for i, i_subject in enumerate([subjects[idx] for idx in subject_idx]):
                    for j, j_subject in enumerate([subjects[idx] for idx in subject_idx]):
                        if i_subject != j_subject:
                            try:
                                spectrum = h5f[f"data/{i_subject}_{j_subject}_{movie}"]["observed"][:MIN_LENGTH]
                            except:
                                spectrum = h5f[f"data/{j_subject}_{i_subject}_{movie}"]["observed"][:MIN_LENGTH]
                            for n_bin, bin_id in enumerate(unique_bins):
                                all_spectra[i, j, n_bin] = spectrum[bin_assignments == bin_id].mean()
                                #all_spectra[j, i, n_bin] = all_spectra[i, j, n_bin]
                bootstrapped_series[movie] = all_spectra
        
        with open(file, "wb") as f:
            pickle.dump((bootstrapped_series, subject_idx), f)

        boot_idm1 = np.nanmean([bootstrapped_series[movie] for movie in even_movies], axis=0)
        boot_idm2 = np.nanmean([bootstrapped_series[movie] for movie in config.movies[1::2]], axis=0)
        boot_idm1 = np.moveaxis(boot_idm1, -1, 0)
        boot_idm2 = np.moveaxis(boot_idm2, -1, 0)
        bootstrapped_even_motion1 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
        bootstrapped_odd_motion1 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
        bootstrapped_even_motion2 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
        bootstrapped_odd_motion2 = np.zeros((int(n_subjects*0.9), int(n_subjects*0.9)))
        for i, i_subject in enumerate([subjects[idx] for idx in subject_idx]):
            for j, j_subject in enumerate([subjects[idx] for idx in subject_idx]):
                if config.subjects.index(i_subject) < config.subjects.index(j_subject):
                    bootstrapped_even_motion1[i, j] = even_motion1[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                    bootstrapped_even_motion1[j, i] = bootstrapped_even_motion1[i, j]
                    bootstrapped_odd_motion1[i, j] = odd_motion1[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                    bootstrapped_odd_motion1[j, i] = bootstrapped_odd_motion1[i, j]
                    bootstrapped_even_motion2[i, j] = even_motion2[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                    bootstrapped_even_motion2[j, i] = bootstrapped_even_motion2[i, j]
                    bootstrapped_odd_motion2[i, j] = odd_motion2[config.subjects.index(i_subject), config.subjects.index(j_subject)]
                    bootstrapped_odd_motion2[j, i] = bootstrapped_odd_motion2[i, j]
                elif config.subjects.index(i_subject) > config.subjects.index(j_subject):
                    bootstrapped_even_motion1[i, j] = even_motion1[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                    bootstrapped_even_motion1[j, i] = bootstrapped_even_motion1[i, j]
                    bootstrapped_odd_motion1[i, j] = odd_motion1[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                    bootstrapped_odd_motion1[j, i] = bootstrapped_odd_motion1[i, j]
                    bootstrapped_even_motion2[i, j] = even_motion2[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                    bootstrapped_even_motion2[j, i] = bootstrapped_even_motion2[i, j]
                    bootstrapped_odd_motion2[i, j] = odd_motion2[config.subjects.index(j_subject), config.subjects.index(i_subject)]
                    bootstrapped_odd_motion2[j, i] = bootstrapped_odd_motion2[i, j]
        if args.isc:
            corr = np.zeros((n_bins))
            for n_bin in range(n_bins):
                _, rx, ry = analysis._fully_partial_correlation(
                    get_filtered_upper_triangle(boot_idm1[n_bin], subject_idx),
                    get_filtered_upper_triangle(boot_idm2[n_bin], subject_idx),
                    get_filtered_upper_triangle(bootstrapped_even_motion1, subject_idx),
                    get_filtered_upper_triangle(bootstrapped_odd_motion1, subject_idx))
                _, rx, ry = analysis._fully_partial_correlation(
                    rx,
                    ry,
                    get_filtered_upper_triangle(bootstrapped_even_motion2, subject_idx),
                    get_filtered_upper_triangle(bootstrapped_odd_motion2, subject_idx))
                r, _, _ = analysis._fully_partial_correlation(
                    rx,
                    ry,
                    get_filtered_upper_triangle(bootstrapped_even_anat, subject_idx),
                    get_filtered_upper_triangle(bootstrapped_odd_anat, subject_idx))
                corr[n_bin] = r
        else:
            corr = np.zeros((n_bins))
            for n_bin in range(n_bins):
                _, rx, ry = analysis._fully_partial_correlation(
                    get_filtered_upper_triangle(boot_idm1[n_bin], subject_idx),
                    get_filtered_upper_triangle(boot_idm2[n_bin], subject_idx),
                    get_filtered_upper_triangle(bootstrapped_even_motion1, subject_idx),
                    get_filtered_upper_triangle(bootstrapped_odd_motion1, subject_idx))
                r, rx, ry = analysis._fully_partial_correlation(
                    rx,
                    ry,
                    get_filtered_upper_triangle(bootstrapped_even_motion2, subject_idx),
                    get_filtered_upper_triangle(bootstrapped_odd_motion2, subject_idx))
                corr[n_bin] = r
        bootstrapped_correlations.append(corr)    
    return bootstrapped_correlations
    

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

    roi_boot_corrs = {roi: [] for roi in rois}
    roi_boot_corrs_isc = {roi: [] for roi in rois}

    for roi in rois:
        args = argparse.Namespace(
            dir=f"cov_{roi}_functional_perm",
            roi=f"{roi}",
            metric="cov",
            motion=True,
            motion_dir="/home/chan21/_individual_dim/results/",
            motion_type=["ISC", "AnnaK"],
            isc=False,
            isc_dir=f"/data/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5",#f"/home/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #"/data/chan21/indiv_diff_dim/results/n=39/isc_occipital_functional/eigenspectra.h5",
            anatomical=False,
            #anatomical_dir="/path/to/anatomical_matrices"
        )

        analysis, even_idm, odd_idm, even_anat, odd_anat, motion_matrices = compute_idm_fp(args)
        movies = list(motion_matrices[0].keys())
        even_motion1 = np.nanmean([motion_matrices[0][m] for m in movies[::2]], axis=0)
        odd_motion1 = np.nanmean([motion_matrices[0][m] for m in movies[1::2]], axis=0)
        even_motion2 = np.nanmean([motion_matrices[1][m] for m in movies[::2]], axis=0)
        odd_motion2 = np.nanmean([motion_matrices[1][m] for m in movies[1::2]], axis=0)

        if args.isc:
            n_bins = len(even_idm)
            correlations = []
            residual1 = []
            residual2 = []
            for n_bin in range(n_bins):
                _, rx, ry = analysis._fully_partial_correlation(
                    get_upper_triangle(even_idm[n_bin]),
                    get_upper_triangle(odd_idm[n_bin]),
                    get_upper_triangle(even_motion1),
                    get_upper_triangle(odd_motion1))
                _, rx, ry = analysis._fully_partial_correlation(
                    rx,
                    ry,
                    get_upper_triangle(even_motion2),
                    get_upper_triangle(odd_motion2))
                corr, res1, res2 = analysis._fully_partial_correlation(
                    rx,
                    ry,
                    get_upper_triangle(even_anat),
                    get_upper_triangle(odd_anat))
                correlations.append(corr)
                residual1.append(res1)
                residual2.append(res2)
            roi_corrs_isc[roi] = correlations
            print(roi, correlations)
            diagonals = np.zeros((len(correlations), len(correlations)))
            for i, corr in enumerate(correlations):
                residual1_tri = residual1[i]
                for j, c in enumerate(correlations):
                    diagonals[i, j] = np.corrcoef(residual1_tri, residual2[j])[0, 1]
            roi_split_corrs_isc[roi] = diagonals
        else:
            n_bins = len(even_idm)
            correlations = []
            residual1 = []
            residual2 = []
            for n_bin in range(n_bins):
                # # Visualization
                # idm_e = zscore_matrix(even_idm[n_bin])
                # idm_o = zscore_matrix(odd_idm[n_bin])
                # save_matrix(idm_e, filename=f"idm_even_{n_bin}.svg", style='coolwarm')
                # save_matrix(idm_o, filename=f"idm_odd_{n_bin}.svg", style='coolwarm')
                # if n_bin == 3:
                #     break
                _, rx, ry = analysis._fully_partial_correlation(
                    get_upper_triangle(even_idm[n_bin]),
                    get_upper_triangle(odd_idm[n_bin]),
                    get_upper_triangle(even_motion1),
                    get_upper_triangle(odd_motion1))
                corr, rx, ry = analysis._fully_partial_correlation(
                    rx,
                    ry,
                    get_upper_triangle(even_motion2),
                    get_upper_triangle(odd_motion2))
                correlations.append(corr)
                residual1.append(rx)
                residual2.append(ry)
            roi_corrs[roi] = correlations
            print(roi, correlations)
            diagonals = np.zeros((len(correlations), len(correlations)))
            for i, corr in enumerate(correlations):
                residual1_tri = residual1[i]
                for j, c in enumerate(correlations):
                    diagonals[i, j] = np.corrcoef(residual1_tri, residual2[j])[0, 1]
            roi_split_corrs[roi] = diagonals

    for roi in rois:
        args = argparse.Namespace(
            dir=f"cov_{roi}_functional_perm",
            roi=f"{roi}",
            metric="cov",
            motion=True,
            motion_dir="/home/chan21/_individual_dim/results/",
            motion_type=["ISC", "AnnaK"],
            isc=True,
            isc_dir=f"/data/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5",#f"/home/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #"/data/chan21/indiv_diff_dim/results/n=39/isc_occipital_functional/eigenspectra.h5",
            anatomical=False,
            #anatomical_dir="/path/to/anatomical_matrices"
        )
        bootstrapped_correlations = bootstrap_idm_series(analysis, args, n_bootstraps=1000)
        if args.isc:
            roi_boot_corrs_isc[roi] = bootstrapped_correlations
        else:
            roi_boot_corrs[roi] = bootstrapped_correlations

        args = argparse.Namespace(
            dir=f"cov_{roi}_functional_perm",
            roi=f"{roi}",
            metric="cov",
            motion=True,
            motion_dir="/home/chan21/_individual_dim/results/",
            motion_type=["ISC", "AnnaK"],
            isc=False,
            isc_dir=f"/data/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5",#f"/home/chan21/idim-debug-spectra/results/isc_{roi}_functional/eigenspectra.h5", #"/data/chan21/indiv_diff_dim/results/n=39/isc_occipital_functional/eigenspectra.h5",
            anatomical=False,
            #anatomical_dir="/path/to/anatomical_matrices"
        )
        bootstrapped_correlations = bootstrap_idm_series(analysis, args, n_bootstraps=1000)
        if args.isc:
            roi_boot_corrs_isc[roi] = bootstrapped_correlations
        else:
            roi_boot_corrs[roi] = bootstrapped_correlations
    
    alpha = 0.05
    for roi in rois:
        for n_bin, bin in enumerate(unique_bins):
            if args.isc:
                lower = np.percentile(np.array([sp[n_bin] for sp in roi_boot_corrs_isc[roi]]), alpha/2 * 100)
                upper = np.percentile(np.array([sp[n_bin] for sp in roi_boot_corrs_isc[roi]]), (1-alpha/2) * 100)
                print(f"{roi}: {bin} {round(roi_corrs_isc[roi][n_bin], 3)}; [{round(lower, 3)}, {round(upper, 3)}]")

if __name__ == "__main__":
    main()