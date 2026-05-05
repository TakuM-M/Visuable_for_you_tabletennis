"""
Dataset module

Provides various dataset classes for pose sequence data:
- CSVPoseSequenceDataset: Load from single CSV file
- MemoryPoseSequenceDataset: Load from in-memory numpy arrays
- MultiCSVPoseDataset: Load from multiple CSV files

All datasets inherit from BasePoseSequenceDataset
"""
from src.datasets.base_dataset import BasePoseSequenceDataset, collate_fn
from src.datasets.csv_dataset import CSVPoseSequenceDataset
from src.datasets.memory_dataset import MemoryPoseSequenceDataset
from src.datasets.multi_csv_dataset import MultiCSVPoseDataset
from src.datasets.augmentation import OnlineAugmentor, OnlineAugmentationConfig

__all__ = [
    'BasePoseSequenceDataset',
    'CSVPoseSequenceDataset',
    'MemoryPoseSequenceDataset',
    'MultiCSVPoseDataset',
    'OnlineAugmentor',
    'OnlineAugmentationConfig',
    'collate_fn',
]
