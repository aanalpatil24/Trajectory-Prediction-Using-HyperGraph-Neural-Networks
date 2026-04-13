"""
Tests for Hypergraph operations.
"""
import unittest
import torch
import numpy as np
from src.utils.hypergraph_builder import HypergraphConstructor
from src.models.hypergraph_conv import HypergraphConv, MultiLayerHGNN


class TestHypergraphConstructor(unittest.TestCase):
    """Test Hypergraph construction with DBSCAN."""

    def setUp(self):
        self.constructor = HypergraphConstructor(eps=2.0, min_samples=2)

    def test_simple_clustering(self):
        """Test basic clustering of close agents."""
        # Create 3 agents in a cluster, 1 isolated
        positions = torch.tensor([
            [0.0, 0.0],
            [0.5, 0.5],  # Close to agent 0
            [1.0, 1.0],  # Close to agent 1
            [10.0, 10.0]  # Isolated
        ])

        hyperedge_index, weights, metadata = self.constructor.construct_hypergraph(
            positions, return_metadata=True
        )

        # Should have 2 hyperedges: 1 group + 1 singleton
        self.assertEqual(metadata['num_hyperedges'], 2)
        self.assertEqual(metadata['num_isolated'], 1)

    def test_all_isolated(self):
        """Test when all agents are far apart."""
        positions = torch.tensor([
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0]
        ])

        _, _, metadata = self.constructor.construct_hypergraph(positions)

        # Should have 3 singleton hyperedges
        self.assertEqual(metadata['num_hyperedges'], 3)
        self.assertEqual(metadata['num_isolated'], 3)

    def test_single_group(self):
        """Test when all agents form one group."""
        positions = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 0.5]
        ])

        _, _, metadata = self.constructor.construct_hypergraph(positions)

        # Should have 1 hyperedge containing all agents
        self.assertEqual(metadata['num_hyperedges'], 1)
        self.assertEqual(metadata['avg_group_size'], 4.0)


class TestHypergraphConv(unittest.TestCase):
    """Test Hypergraph Convolution layer."""

    def setUp(self):
        self.in_channels = 16
        self.out_channels = 32
        self.layer = HypergraphConv(self.in_channels, self.out_channels)

    def test_forward_shape(self):
        """Test output shape is correct."""
        num_nodes = 10
        x = torch.randn(num_nodes, self.in_channels)

        # Simple hyperedge: all nodes in one hyperedge
        hyperedge_index = torch.tensor([
            list(range(num_nodes)),  # Node indices
            [0] * num_nodes          # All in hyperedge 0
        ])

        out = self.layer(x, hyperedge_index)

        self.assertEqual(out.shape, (num_nodes, self.out_channels))

    def test_message_passing(self):
        """Test that message passing occurs."""
        # Create 2 nodes, 1 hyperedge
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        hyperedge_index = torch.tensor([[0, 1], [0, 0]])  # Both in hyperedge 0

        out = self.layer(x, hyperedge_index)

        # Output should be different from input (message passing happened)
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))


class TestMultiLayerHGNN(unittest.TestCase):
    """Test multi-layer HGNN."""

    def setUp(self):
        self.hidden_dim = 64
        self.num_layers = 3
        self.model = MultiLayerHGNN(self.hidden_dim, self.num_layers)

    def test_forward(self):
        """Test forward pass through all layers."""
        num_nodes = 8
        x = torch.randn(num_nodes, self.hidden_dim)
        hyperedge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 0, 1, 1, 1, 1]
        ])

        out = self.model(x, hyperedge_index)

        self.assertEqual(out.shape, (num_nodes, self.hidden_dim))


if __name__ == '__main__':
    unittest.main()
