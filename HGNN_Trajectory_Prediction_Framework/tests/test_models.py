"""
Tests for model components.
"""
import unittest
import torch
from src.models.encoder_decoder import TrajectoryEncoder, TrajectoryDecoder
from src.models.seq2seq_hgnn import Seq2SeqHGNN
from src.utils.config import ModelConfig


class TestTrajectoryEncoder(unittest.TestCase):
    """Test Trajectory Encoder."""

    def setUp(self):
        self.config = ModelConfig()
        self.encoder = TrajectoryEncoder(
            input_dim=2,
            hidden_dim=64,
            num_layers=2
        )

    def test_output_shape(self):
        """Test encoder output shape."""
        batch_size = 4
        obs_len = 8
        num_agents = 10

        trajectory = torch.randn(batch_size, obs_len, num_agents, 2)
        hidden = self.encoder(trajectory)

        self.assertEqual(hidden.shape, (batch_size, num_agents, 64))


class TestTrajectoryDecoder(unittest.TestCase):
    """Test Trajectory Decoder."""

    def setUp(self):
        self.config = ModelConfig()
        self.decoder = TrajectoryDecoder(
            output_dim=2,
            hidden_dim=64,
            num_layers=2,
            pred_len=12
        )

    def test_prediction_shape(self):
        """Test decoder output shape."""
        batch_size = 4
        num_agents = 10

        social_features = torch.randn(batch_size, num_agents, 64)
        last_pos = torch.randn(batch_size, num_agents, 2)

        predictions = self.decoder(social_features, last_pos)

        self.assertEqual(predictions.shape, (batch_size, 12, num_agents, 2))


class TestSeq2SeqHGNN(unittest.TestCase):
    """Test complete Seq2Seq HGNN model."""

    def setUp(self):
        self.config = ModelConfig(
            hidden_dim=32,
            num_hgnn_layers=1,
            obs_len=8,
            pred_len=12,
            device='cpu'
        )
        self.model = Seq2SeqHGNN(self.config)

    def test_forward_pass(self):
        """Test full forward pass."""
        batch_size = 2
        obs_len = 8
        pred_len = 12
        num_agents = 5

        obs = torch.randn(batch_size, obs_len, num_agents, 2)
        pred = torch.randn(batch_size, pred_len, num_agents, 2)

        output = self.model(obs, ground_truth=pred, teacher_forcing_ratio=0.5)

        self.assertIn('predictions', output)
        self.assertEqual(output['predictions'].shape, (batch_size, pred_len, num_agents, 2))

    def test_inference_mode(self):
        """Test inference without ground truth."""
        batch_size = 2
        obs = torch.randn(batch_size, 8, 5, 2)

        output = self.model.predict(obs)

        self.assertEqual(output['predictions'].shape, (batch_size, 12, 5, 2))


if __name__ == '__main__':
    unittest.main()
