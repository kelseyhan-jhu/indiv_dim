from pathlib import Path
import pickle
import hashlib
from typing import Optional, Any, Callable, Tuple, List
import numpy as np
import logging
from functools import wraps
from src.utils._config import Movie
import h5py
import shutil

def cached(key_fn: Optional[Callable] = None):
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'cache'):
                logging.warning(f"No cache attribute found on {self.__class__.__name__}")
                return method(self, *args, **kwargs)
            if self.cache is None:
                logging.warning(f"Cache is None on {self.__class__.__name__}")
                return method(self, *args, **kwargs)
            
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default key generation
                try:
                    cache_key = f"{method.__name__}_{kwargs['roi']}_{kwargs['metric']}_{str(kwargs['perm_idx'])}_{','.join(kwargs['movies'])}" \
                    if 'perm_idx' in kwargs else f"{method.__name__}_{kwargs['roi']}_{kwargs['metric']}_{','.join(kwargs['movies'])}"
                except KeyError as e:
                    raise ValueError(f"Missing required parameter: {e}")
            # Check cache
            result = self.cache.get(cache_key)
            if result is not None:
                return result
                
            # Compute and cache result
            result = method(self, *args, **kwargs)
            self.cache.set(cache_key, result)
            return result
            
        return wrapper
    return decorator


class DataCache:
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Cache initialized: enabled={enabled}, dir={cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            logging.debug(f"Cache disabled, skipping get for key: {key}")
            return None
            
        path = self.cache_dir / f"{key}.pkl"
        logging.debug(f"Attempting to read cache: {path}")
        
        try:
            if path.exists():
                with open(path, 'rb') as f:
                    result = pickle.load(f)
                    logging.debug(f"Cache hit for key: {key}")
                    return result
            logging.debug(f"Cache miss for key: {key}")
        except Exception as e:
            logging.warning(f"Cache read failed for {key}: {e}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            logging.debug(f"Cache disabled, skipping set for key: {key}")
            return
            
        path = self.cache_dir / f"{key}.pkl"
        logging.debug(f"Attempting to write cache: {path}")
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(value, f)
                logging.debug(f"Cache written for key: {key}")
        except Exception as e:
            logging.warning(f"Cache write failed for {key}: {e}")


class SpectraAggregator:
    """Aggregate spectra from multiple H5 files into a single comprehensive file."""
    
    # def __init__(self, analysis_dir: Path):
    #     self.analysis_dir = Path(analysis_dir)
    #     self.output_path = self.analysis_dir / "eigenspectra.h5"
    def __init__(self, input_path: Path, output_path: Path):
        self.analysis_dir = Path(input_path)
        self.output_path = Path(output_path) / "eigenspectra.h5"
          
    def _get_input_files(self) -> List[Path]:
        """Get all eigenspectra H5 files in the analysis directory."""
        return list(self.analysis_dir.glob("eigenspectra_*.h5"))
        
    def _copy_attributes(self, source_h5: h5py.File, dest_h5: h5py.File) -> None:
        """Copy attributes from source to destination H5 file."""
        for key, value in source_h5.attrs.items():
            if key not in dest_h5.attrs:
                dest_h5.attrs[key] = value
                
    def _combine_h5_files(self) -> None:
        """Combine multiple H5 files into a single comprehensive file."""
        input_files = self._get_input_files()
        if not input_files:
            raise FileNotFoundError(f"No eigenspectra files found in {self.analysis_dir}")
            
        with h5py.File(self.output_path, 'w') as dest_h5:
            # Copy data from each input file
            for input_path in input_files:
                with h5py.File(input_path, 'r') as source_h5:
                    # Copy attributes from first file
                    if input_path == input_files[0]:
                        self._copy_attributes(source_h5, dest_h5)
                    
                    # Copy all datasets
                    for name, dataset in source_h5['data'].items():
                        if name not in dest_h5:
                            dest_h5.create_group(f'data/{name}')
                        for key in dataset.keys():
                            if key not in dest_h5[f'data/{name}']:
                                source_h5.copy(f'data/{name}/{key}', 
                                             dest_h5[f'data/{name}']) 
                                
    def ensure_combined_file(self) -> Path:
        """Ensure a combined eigenspectra file exists, creating it if necessary."""
        if not self.output_path.exists():
            self._combine_h5_files()
        return self.output_path