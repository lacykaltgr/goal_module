import os
import torch
import numpy as np
from torch.utils.data import Dataset


class DummyGoalDataset(Dataset):
    """
    Dummy dataset for testing the pipeline with goal-oriented data structure.

    Returns:
        - features: [N_OBSERVATIONS, 1024] tensor
        - goal_features: [1, 1024] tensor
        - actions: [8, 3] tensor
        - goal_position: [3] tensor containing goal_x, goal_y, goal_yaw
        - dataset_index: [1] tensor
    """

    def __init__(self, train=True, n_samples=1000):
        self.train = train
        self.n_samples = n_samples

        # Generate dummy data
        self._generate_dummy_data()

    def _generate_dummy_data(self):
        """Generate dummy data with the specified shapes."""
        # Set random seed for reproducibility
        torch.manual_seed(42 if self.train else 123)
        np.random.seed(42 if self.train else 123)

        # Generate features for each sample [N_SAMPLES, N_OBSERVATIONS, 1024]
        # 2 is a hardcoded value, just random
        self.features = torch.randn(self.n_samples, 2, 1024)

        # Generate goal features for each sample [N_SAMPLES, 1, 1024]
        self.goal_features = torch.randn(self.n_samples, 1, 1024)

        # Generate actions for each sample [N_SAMPLES, 8, 3]
        self.actions = torch.randn(self.n_samples, 8, 3)

        # Generate goal positions [N_SAMPLES, 3] (goal_x, goal_y, goal_yaw)
        self.goal_positions = torch.randn(self.n_samples, 3)

        # Generate dataset indices [N_SAMPLES, 1]
        self.dataset_indices = torch.arange(self.n_samples).unsqueeze(1).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, goal_features, actions, goal_position, dataset_index)
        """
        current_features = self.features[index][0]
        goal_features = self.goal_features[index][0]
        features = torch.cat([current_features, goal_features], dim=0)
        
        return (
            features,   # [N_OBSERVATIONS, 1024]
            self.goal_positions[index],             # [3]
        )

    def __len__(self):
        return self.n_samples