"""
Tests for data augmentation.
"""
import unittest
import torch
import numpy as np
from src.data.augmentation import TrajectoryAugmenter
from src.utils.config import ModelConfig


class TestTrajectoryAugmenter(unittest.TestCase):
    """Test data augmentation strategies."""

    def setUp(self):
        self.config = ModelConfig(
            augmentation_enabled=True,
            augmentation_prob=1.0,  # Always apply
            reverse_prob=1.0,
            missing_data_prob=0.1
        )
        self.augmenter = TrajectoryAugmenter(self.config)

        # Create sample trajectory
        self.trajectory = torch.tensor([
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.1, 0.1], [1.1, 1.1]],
            [[0.2, 0.2], [1.2, 1.2]],
        ])  # (3, 2, 2)

    def test_reverse_trajectory(self):
        """Test trajectory reversal."""
        reversed_traj = self.augmenter.reverse_trajectory(self.trajectory)

        # First frame of reversed should equal last frame of original
        self.assertTrue(torch.allclose(reversed_traj[0], self.trajectory[-1]))
        self.assertTrue(torch.allclose(reversed_traj[-1], self.trajectory[0]))

    def test_missing_data_imputation(self):
        """Test missing data injection and imputation."""
        corrupted, mask = self.augmenter.inject_missing_data(
            self.trajectory, 
            missing_prob=0.3
        )

        # Check no NaNs after imputation
        self.assertFalse(torch.isnan(corrupted).any())

        # Check mask shape
        self.assertEqual(mask.shape, (3, 2, 2))

    def test_gaussian_noise(self):
        """Test noise injection."""
        noisy = self.augmenter.add_gaussian_noise(self.trajectory, noise_scale=0.01)

        # Should be different from original
        self.assertFalse(torch.allclose(noisy, self.trajectory))

        # Should be close (small noise)
        self.assertTrue(torch.allclose(noisy, self.trajectory, atol=0.1))


if __name__ == '__main__':
    unittest.main()
