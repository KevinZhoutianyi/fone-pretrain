"""
Data mixture loading for streaming datasets with sampling weights.

Supports two config formats:

1) Local/paths format (legacy):
{
  "datasets": [
    {"name": "math", "path": "/path/to/*.jsonl", "weight": 0.3},
    ...
  ],
  "shuffle_buffer_size": 10000,
  "num_workers": 4
}

2) Hugging Face streaming format (recommended):
{
  "streams": [
    {"name": "fineweb", "hf": "HuggingFaceFW/fineweb", "split": "train", "weight": 0.65},
    {"name": "openwebmath", "hf": "open-web-math/open-web-math", "split": "train", "weight": 0.15},
    {"name": "proofpile2", "hf": "EleutherAI/proof-pile-2", "split": "train", "weight": 0.10},
    {"name": "stackv2", "hf": "bigcode/the-stack-v2-dedup", "split": "train", "weight": 0.10}
  ],
  "shuffle_buffer_size": 10000,
  "num_workers": 4
}
"""

import json
import os
import random
import glob
from typing import Dict, List, Iterator, Optional, Any
from datasets import Dataset, IterableDataset, load_dataset, concatenate_datasets
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataMixture:
    """
    Streaming data mixture loader that samples from multiple datasets according to weights.
    """
    
    def __init__(
        self,
        mixture_config: str,
        sequence_length: int = 2048,
        shuffle_buffer_size: int = 10000,
        num_workers: int = 4,
        seed: int = 42
    ):
        """
        Initialize data mixture.
        
        Args:
            mixture_config: Path to mixture configuration JSON
            sequence_length: Target sequence length for packing
            shuffle_buffer_size: Buffer size for shuffling
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
        """
        self.mixture_config = mixture_config
        self.sequence_length = sequence_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Load mixture configuration
        with open(mixture_config, 'r') as f:
            self.config = json.load(f)
            
        # Determine config mode
        self.streams_info = self.config.get('streams')
        self.datasets_info = self.config.get('datasets')
        self.total_tokens_target = self.config.get('total_tokens_target', 30_000_000_000)
        
        # Normalize weights
        if self.streams_info:
            total_weight = sum(s['weight'] for s in self.streams_info)
            for s in self.streams_info:
                s['normalized_weight'] = s['weight'] / total_weight
        elif self.datasets_info:
            total_weight = sum(ds['weight'] for ds in self.datasets_info)
            for ds in self.datasets_info:
                ds['normalized_weight'] = ds['weight'] / total_weight
        else:
            raise ValueError("Mixture config must contain either 'streams' or 'datasets'")
            
        if self.streams_info:
            logger.info(f"Loaded mixture config with {len(self.streams_info)} streams (HF streaming)")
            for s in self.streams_info:
                logger.info(f"  {s['name']}: hf={s['hf']}, split={s.get('split','train')}, weight={s['weight']:.3f} ({s['normalized_weight']:.3f})")
        else:
            logger.info(f"Loaded mixture config with {len(self.datasets_info)} datasets (local/paths)")
            for ds in self.datasets_info:
                logger.info(f"  {ds['name']}: path={ds['path']}, weight={ds['weight']:.3f} ({ds['normalized_weight']:.3f})")
            
        # Initialize datasets
        self.datasets = {}
        self.dataset_iterators = {}
        
        # Set up random state
        self.rng = random.Random(seed)
        
    def _load_dataset_from_path(self, path: str, name: str) -> Dataset:
        """Load dataset from file path(s)."""
        path_obj = Path(path)
        
        if '*' in path:
            # Handle glob patterns
            files = glob.glob(path)
            if not files:
                raise ValueError(f"No files found matching pattern: {path}")
                
            logger.info(f"Loading {len(files)} files for dataset {name}")
            
            # Load and concatenate all files
            datasets = []
            for file_path in sorted(files):
                try:
                    if file_path.endswith('.jsonl'):
                        ds = load_dataset('json', data_files=file_path, split='train')
                    elif file_path.endswith('.json'):
                        ds = load_dataset('json', data_files=file_path, split='train')
                    elif file_path.endswith('.txt'):
                        ds = load_dataset('text', data_files=file_path, split='train')
                    else:
                        logger.warning(f"Unknown file format: {file_path}")
                        continue
                        
                    datasets.append(ds)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
                    
            if not datasets:
                raise ValueError(f"No valid datasets loaded for {name}")
                
            return concatenate_datasets(datasets)
            
        else:
            # Single file or directory
            if path_obj.suffix == '.jsonl':
                return load_dataset('json', data_files=path, split='train')
            elif path_obj.suffix == '.json':
                return load_dataset('json', data_files=path, split='train')
            elif path_obj.suffix == '.txt':
                return load_dataset('text', data_files=path, split='train')
            else:
                # Try to load as a HuggingFace dataset
                return load_dataset(path, split='train')
    
    def _load_stream_from_hf(self, hf_name: str, split: str, name: str, config: Optional[str] = None) -> IterableDataset:
        """Load Hugging Face streaming dataset (IterableDataset)."""
        # Avoid contamination: do not allow GSM8K in pretraining streams
        if 'gsm8k' in hf_name.lower():
            raise ValueError("GSM8K must not be included in pretraining mixture (to avoid contamination)")
        cfg_msg = f", config={config}" if config else ""
        logger.info(f"Loading HF streaming dataset: {hf_name} (split={split}{cfg_msg}) for stream {name}")
        # Some datasets require a config name and/or trust_remote_code
        # Use work directory for caching to avoid filling up HOME
        cache_dir = os.environ.get('WORK_HDD_DIR', None)
        if cache_dir:
            cache_dir = os.path.join(cache_dir, '.cache', 'huggingface', 'datasets')
        
        load_kwargs = dict(streaming=True, split=split, cache_dir=cache_dir)
        
        # First try without trust_remote_code
        try:
            if config:
                ds = load_dataset(hf_name, config, **load_kwargs)
            else:
                ds = load_dataset(hf_name, **load_kwargs)
            # Shuffle streaming iterator with buffer
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
            return ds
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                # Try with trust_remote_code=True for legacy datasets
                logger.warning(f"Attempting to load {name} ({hf_name}) with trust_remote_code=True")
                load_kwargs['trust_remote_code'] = True
                try:
                    if config:
                        ds = load_dataset(hf_name, config, **load_kwargs)
                    else:
                        ds = load_dataset(hf_name, **load_kwargs)
                    # Shuffle streaming iterator with buffer
                    ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
                    return ds
                except Exception as e2:
                    logger.error(f"Failed to load stream {name} ({hf_name}) even with trust_remote_code=True: {e2}")
                    raise
            else:
                logger.error(f"Failed to load stream {name} ({hf_name}): {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to load stream {name} ({hf_name}): {e}")
            raise
                
    def _initialize_datasets(self):
        """Initialize all datasets and their iterators."""
        if self.streams_info:
            # HF streaming mode
            for s in self.streams_info:
                name = s['name']
                hf_name = s['hf']
                split = s.get('split', 'train')
                config = s.get('config')
                try:
                    dataset = self._load_stream_from_hf(hf_name, split, name, config)
                    self.datasets[name] = dataset
                    extras = f", config={config}" if config else ""
                    logger.info(f"Loaded streaming dataset {name} (HF: {hf_name}{extras})")
                except Exception as e:
                    logger.error(f"Failed to load stream {name} ({hf_name}): {e}")
                    raise
        else:
            # Local/paths mode
            for ds_info in self.datasets_info:
                name = ds_info['name']
                path = ds_info['path']
                
                try:
                    logger.info(f"Loading dataset: {name}")
                    dataset = self._load_dataset_from_path(path, name)
                    
                    # Shuffle dataset
                    dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
                    
                    self.datasets[name] = dataset
                    
                    logger.info(f"Loaded dataset {name} with {len(dataset)} examples")
                    
                except Exception as e:
                    logger.error(f"Failed to load dataset {name} from {path}: {e}")
                    raise
                
    def _get_dataset_iterator(self, name: str) -> Iterator[Dict[str, Any]]:
        """Get iterator for a specific dataset with infinite cycling."""
        dataset = self.datasets[name]
        
        while True:
            if isinstance(dataset, IterableDataset):
                # Streaming dataset: can re-shuffle with a new seed
                shuffled_dataset = dataset.shuffle(seed=self.rng.randint(0, 2**32-1), buffer_size=self.shuffle_buffer_size)
                for example in shuffled_dataset:
                    yield example
            else:
                # Map-style dataset
                shuffled_dataset = dataset.shuffle(seed=self.rng.randint(0, 2**32-1), buffer_size=self.shuffle_buffer_size)
                for example in shuffled_dataset:
                    yield example
                
    def _extract_text(self, example: Dict[str, Any]) -> str:
        """Extract text content from an example."""
        # Common text field names
        text_fields = ['text', 'content', 'input', 'prompt', 'question', 'problem']
        
        for field in text_fields:
            if field in example and example[field]:
                return str(example[field])
                
        # If no standard field found, try to find any string field
        for key, value in example.items():
            if isinstance(value, str) and value.strip():
                return value
                
        logger.warning(f"No text content found in example: {list(example.keys())}")
        return ""
        
    def get_streaming_iterator(self) -> Iterator[str]:
        """
        Get streaming iterator over mixed text data.
        
        Yields:
            Text strings from the mixed datasets
        """
        # Initialize datasets if not already done
        if not self.datasets:
            self._initialize_datasets()
            
        # Initialize dataset iterators
        for name in self.datasets.keys():
            self.dataset_iterators[name] = self._get_dataset_iterator(name)
            
        # Create weighted sampling
        dataset_names = list(self.datasets.keys())
        if self.streams_info:
            weights = [s['normalized_weight'] for s in self.streams_info]
        else:
            weights = [ds['normalized_weight'] for ds in self.datasets_info]
        
        logger.info("Starting streaming data mixture")
        
        while True:
            # Sample dataset according to weights
            chosen_dataset = self.rng.choices(dataset_names, weights=weights, k=1)[0]
            
            # Get next example from chosen dataset
            try:
                example = next(self.dataset_iterators[chosen_dataset])
                text = self._extract_text(example)
                
                if text.strip():  # Only yield non-empty text
                    yield text
                    
            except StopIteration:
                # This shouldn't happen with infinite cycling, but just in case
                logger.warning(f"Iterator exhausted for dataset {chosen_dataset}")
                self.dataset_iterators[chosen_dataset] = self._get_dataset_iterator(chosen_dataset)
                continue
                
            except Exception as e:
                logger.warning(f"Error processing example from {chosen_dataset}: {e}")
                continue
                
    def estimate_dataset_sizes(self) -> Dict[str, int]:
        """Estimate the size of each dataset in the mixture."""
        if not self.datasets:
            self._initialize_datasets()
            
        sizes = {}
        for name, dataset in self.datasets.items():
            try:
                sizes[name] = len(dataset)  # Map-style datasets only
            except TypeError:
                sizes[name] = -1  # Unknown for streaming datasets
            
        return sizes
        
    def get_mixture_stats(self) -> Dict[str, Any]:
        """Get statistics about the data mixture."""
        sizes = self.estimate_dataset_sizes()
        
        if self.streams_info:
            weights = {s['name']: s['normalized_weight'] for s in self.streams_info}
            total_sources = len(self.streams_info)
        else:
            weights = {ds['name']: ds['normalized_weight'] for ds in self.datasets_info}
            total_sources = len(self.datasets_info)
        
        stats = {
            'total_sources': total_sources,
            'dataset_sizes': sizes,
            'weights': weights,
            'streaming': bool(self.streams_info is not None),
            'target_tokens': self.total_tokens_target,
            'sequence_length': self.sequence_length
        }
        
        return stats


def load_data_mixture(config_path: str, **kwargs) -> DataMixture:
    """
    Convenience function to load a data mixture from configuration.
    
    Args:
        config_path: Path to mixture configuration JSON
        **kwargs: Additional arguments for DataMixture
        
    Returns:
        DataMixture instance
    """
    return DataMixture(config_path, **kwargs)


def create_example_mixture_config(
    output_path: str,
    datasets: List[Dict[str, Any]],
    total_tokens: int = 30_000_000_000,
    sequence_length: int = 2048
):
    """
    Create an example mixture configuration file.
    
    Args:
        output_path: Where to save the configuration
        datasets: List of dataset configurations
        total_tokens: Target total tokens
        sequence_length: Sequence length for packing
    """
    config = {
        "datasets": datasets,
        "total_tokens_target": total_tokens,
        "sequence_length": sequence_length,
        "shuffle_buffer_size": 10000,
        "num_workers": 4
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Created example mixture config at {output_path}")


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
        # Test the mixture
        mixture = DataMixture(config_path)
        
        # Print stats
        stats = mixture.get_mixture_stats()
        print("Mixture statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        # Test streaming
        print("\nTesting streaming (first 5 examples):")
        stream = mixture.get_streaming_iterator()
        
        for i, text in enumerate(stream):
            if i >= 5:
                break
            print(f"Example {i+1}: {text[:100]}...")
            
    else:
        print("Usage: python mixture.py <config_path>")
        print("Example: python mixture.py configs/data_mixture.example.json")
