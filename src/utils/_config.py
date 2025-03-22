from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from enum import Enum, auto
from typing import List, Union
from dataclasses import dataclass
from typing import List
import logging
import sys


class Movie(str, Enum):
    iteration = auto()
    defeat = auto()
    growth = auto()
    lemonade = auto()


class Subjects:
    subjects: List[str] = [
        "sub-NSD103",
        "sub-NSD104",
        "sub-NSD105",
        "sub-NSD106",
        "sub-NSD107",
        "sub-NSD108",
        "sub-NSD109",
        "sub-NSD110",
        "sub-NSD111",
        "sub-NSD112",
        "sub-NSD113",
        "sub-NSD114",
        "sub-NSD115",
        "sub-NSD116",
        "sub-NSD117",
        "sub-NSD119",
        "sub-NSD120",
        "sub-NSD122",
        "sub-NSD123",
        "sub-NSD124",
        "sub-NSD125",
        "sub-NSD126",
        "sub-NSD127",
        "sub-NSD128",
        "sub-NSD129",
        "sub-NSD130",
        "sub-NSD132",
        "sub-NSD133",
        "sub-NSD134",
        "sub-NSD135",
        "sub-NSD136",
        "sub-NSD138",
        "sub-NSD140",
        "sub-NSD142",
        "sub-NSD145",
        "sub-NSD146",
        "sub-NSD147",
        "sub-NSD148",
        "sub-NSD149",
        "sub-NSD150",
        "sub-NSD151",
        "sub-NSD153",
        "sub-NSD155"
    ]

    @classmethod
    def exclude_subjects(cls, subjects_to_exclude: List[Union[int, str, 'Subjects']]) -> List[str]:
        """
        Create a filtered subject list by excluding subjects specified by 
        indices, subject codes, or enum members
        """
        all_subjects = cls.subjects
        excluded_values = set()
        
        for subject in subjects_to_exclude:
            excluded_values.add(subject)
                
        return [str(subj) for subj in all_subjects 
                if str(subj) not in excluded_values]

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> List[str]:
        """Load subjects from a YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Assuming the YAML has a 'subjects' key
        subjects = config.get('subjects', [])
        
        # Validate that all subjects in YAML exist in enum
        valid_subjects = set(str(s) for s in cls)
        for subject in subjects:
            if subject not in valid_subjects:
                raise ValueError(f"Invalid subject in YAML: {subject}")
                
        return subjects

    @classmethod
    def get_analysis_subjects(cls, 
                            yaml_path: Union[str, Path, None] = None,
                            exclude_subjects: List[Union[int, str, 'Subjects']] = None) -> List[str]:
        """
        Flexible method to get subjects either from YAML or by exclusion
        """
        if yaml_path is not None:
            return cls.from_yaml(yaml_path)
        elif exclude_subjects is not None:
            return cls.exclude_subjects(exclude_subjects)
        else:
            return cls.subjects

@dataclass
class AnalysisConfig:
    subjects: Subjects
    movies: List[Movie]
    data_path: Path
    roi_names: List[str]
    metric: str
    alignment: str
    n_downsample: int
    n_permutations: int
    random_state: int
    perform_downsampling: bool = False
    perform_permutations: bool = False
    overwrite: bool = False
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    block_size: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Path) -> 'AnalysisConfig':
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            if not 'data_path' in config_dict:
                config_dict['data_path'] = None
        return cls(**config_dict)
    

class Config:
    subjects: List[str]
    movies: List[Movie]
    metric: str
    analysis_name: str
    other_params: dict = None  # For any additional parameters
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Create Config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Load subjects using Subjects class
        if 'subjects' in config_dict:
            subjects = config_dict['subjects']
        else:
            subjects = [str(s) for s in Subjects]  # All subjects if not specified
            
        # Validate and convert movies to Movie enum
        movies = [
            Movie(movie) if isinstance(movie, str) else movie 
            for movie in config_dict.get('movies', list(Movie))
        ]
        
        # Get other parameters with defaults
        metric = config_dict.get('metric', 'correlation')
        analysis_name = config_dict.get('analysis_name', 'default_analysis')
        
        # Store any additional parameters
        other_params = {
            k: v for k, v in config_dict.items() 
            if k not in {'subjects', 'movies', 'metric', 'analysis_name'}
        }
        
        return cls(
            subjects=subjects,
            movies=movies,
            metric=metric,
            analysis_name=analysis_name,
            other_params=other_params
        )
    
    @classmethod
    def create(cls,
               subjects: Union[List[str], str, Path, None] = None,
               movies: List[Union[str, Movie]] = None,
               metric: str = 'correlation',
               analysis_name: str = 'default_analysis',
               exclude_subjects: List[Union[int, str, Subjects]] = None,
               **kwargs) -> 'Config':
        """
        Create Config manually or from combination of parameters
        
        Args:
            subjects: List of subjects, path to YAML, or None for all subjects
            movies: List of movies (strings or Movie enum members)
            metric: Eigenspectra metric
            analysis_name: Name of the analysis
            exclude_subjects: Subjects to exclude if using full subject list
            **kwargs: Additional parameters
            
        Usage examples:
        # 1. Load from YAML
        config1 = Config.from_yaml('analysis_config.yaml')

        #2. Create manually with specific parameters
        config2 = Config.create(
            exclude_subjects=[1, 5, 8],
            movies=[Movie.ITERATION, Movie.GROWTH],
            metric='correlation',
            analysis_name='movie_analysis',
            smoothing_window=5  # Additional parameter
        )

        """
        # Handle subjects
        if isinstance(subjects, (str, Path)):
            # If path provided, load from YAML
            return cls.from_yaml(subjects)
        elif subjects is None:
            # If None, get all subjects or exclude specified ones
            subjects = Subjects.get_analysis_subjects(
                exclude_subjects=exclude_subjects
            )
        
        # Handle movies
        if movies is None:
            movies = list(Movie)
        else:
            movies = [Movie(m) if isinstance(m, str) else m for m in movies]
            
        return cls(
            subjects=subjects,
            movies=movies,
            metric=metric,
            analysis_name=analysis_name,
            other_params=kwargs
        )
    
