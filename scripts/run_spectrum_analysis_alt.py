# Alternate permutations for per subject pair

import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils._config import *
from src.utils._io import *
from src.utils._paths import *
from src.spectra._definition import *

import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run spectral analysis")
    parser.add_argument("--dir", type=str, help="Directory containing .h5 and .yaml")
    parser.add_argument("--roi", type=str)
    parser.add_argument("--metric", type=str, default="cov",
                        choices=["cov", "corr", "isc"])
    parser.add_argument("--alignment", type=str, default="functional",
                        choices=["functional", "anatomical"])
    parser.add_argument("--perform_permutations", action="store_true")
    parser.add_argument("--n_permutations", type=int, default=1000)
    return parser.parse_args()

def main():    
    # Simulate arguments
    args = argparse.Namespace(
        dir="cov_pmc_functional_perm",
        roi="pmc",
        metric="cov",
        alignment="functional",
        perform_permutations=True,
        n_permutations=1000
        )
    
    # Initialize configuration
    ANALYSIS_DIR = Path("/data/chan21/idim-debug-spectra/results") / args.dir  #Path("/home/chan21/indiv_dim/results") / args.dir #HOME_DIR / "results" / args.dir
    config = AnalysisConfig.from_yaml(ANALYSIS_DIR / "config.yaml")

    # Ensure combined eigenspectra file exists
    if not (ANALYSIS_DIR / "eigenspectra.h5").exists():
        aggregator = SpectraAggregator(input_path=ANALYSIS_DIR, #/home/chan21/idim-debug-spectra/results_odd/cov_occipital_functional_odd/",
                                       output_path=ANALYSIS_DIR)
        combined_h5_path = aggregator.ensure_combined_file()
        config.data_path = combined_h5_path
    else:
        config.data_path = ANALYSIS_DIR / "eigenspectra.h5"

    assert config.roi_names[0] == args.roi
    assert config.metric == args.metric
    assert config.alignment == args.alignment
    assert config.perform_permutations == args.perform_permutations
    assert config.n_permutations == args.n_permutations

    # Define bin assignments
    #indices = np.arange(719) if args.roi == "calcarine" else np.arange(MIN_LENGTH)
    # bin_assignments = [assign_data_to_geometrically_spaced_bins(
    #     index + 1,
    #     density=3,
    #     start=1,
    #     stop=10_000) for index in indices]
    # unique_bins = np.unique(bin_assignments)
    
    # Generate spectra
    generator = SpectraGenerator(config)
    observed_spectra = list(
        generator.yield_spectra()
        )

    # Create processor
    cache = DataCache(
        cache_dir = HOME_DIR / "data" / "cache",
        enabled=True,
        )
    processor = SpectraProcessor(cache=cache)

    # Define bin assignments
    min_length = processor.get_min_length(observed_spectra)
    indices = np.arange(min_length)
    bin_assignments = [assign_data_to_geometrically_spaced_bins(
        index + 1,
        density=3,
        start=1,
        stop=10_000) for index in indices]
    unique_bins = np.unique(bin_assignments)

    # Process spectra
    averaged_observed = processor.average_spectra(
        spectra=observed_spectra,
        roi = config.roi_names[0],
        metric = config.metric,
        movies = config.movies,
        permuted=False,
        length=min_length
        )
    
    binned_observed = processor.bin_spectra(
        averaged_observed['means'],
        roi = config.roi_names[0],
        metric = config.metric,
        movies = config.movies,
        permuted=False
        )

    # Process permuted spectra if enabled
    if config.perform_permutations:

        # Create processor
        cache_perm = DataCache(
            #cache_dir = HOME_DIR / "data" / "cache_perm",
            cache_dir = "/data/chan21/idim-debug-spectra/data/cache_perm",
            enabled=True,
            )
        processor = SpectraProcessor(cache=cache_perm)


        subject_pair_list = [(subject1, subject2) for i, subject1 in enumerate(config.subjects) for subject2 in config.subjects[i+1:]]
        subject_pair_dict = {
            (subject1, subject2): np.zeros((config.n_permutations, len(unique_bins))) for (subject1, subject2) in subject_pair_list
        }
        for perm_idx in tqdm(range(config.n_permutations)):
            if (perm_idx + 1) % 100 == 0:
                print(f"Processed {perm_idx + 1}/{config.n_permutations} permutations")
                
            file = f"{cache_perm.cache_dir}/bin_pairwise_{args.roi}_{args.metric}_{str(perm_idx)}_{','.join(config.movies)}.pkl"
            if os.path.exists(file):
                continue
            for subject1, subject2 in subject_pair_list:
                spectra = np.zeros((len(config.movies), len(unique_bins)))
                for n_mov, movie in enumerate(config.movies):
                    kwargs = {
                        'subject_pair': (subject1, subject2),
                        'perm_idx' : perm_idx,
                    }
                    with h5py.File(config.data_path, 'r') as h5f:
                        spectrum = h5f[f"data/{kwargs['subject_pair'][0]}_{kwargs['subject_pair'][1]}_{movie}"]['permutations'][kwargs['perm_idx']]
                    binned_permuted = average_bin(spectrum[:min_length], bin_assignments)
                    spectra[n_mov] = binned_permuted.means
                    subject_pair_dict[(subject1, subject2)][perm_idx] = spectra.mean(axis=0)
            
            with open(file, 'wb') as f:
                pickle.dump(subject_pair_dict, f)  
            

if __name__ == "__main__":
    main()