"""Tests for the `optimize.py` module."""
from copy import deepcopy

import pytest
import torch
from torch_geometric.nn import NNConv, global_mean_pool

from gnnfit.convert import Linear
from gnnfit.optimize import gradient_wrt_input


class Net(torch.nn.Module):
    def __init__(self, n_channels=2, hidden_size=256, n_targets=1):
        super(Net, self).__init__()

        self.conv = NNConv(1, n_channels, nn=torch.nn.Linear(1, n_channels))
        self.fc1 = torch.nn.Linear(n_channels, hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(hidden_size, n_targets, bias=True)

    def forward(self, data):
        """Run a forward pass.

        We do not use any activations just ot make sure an untrained
        netwrok will be still able to propagate gradients.

        Parameters
        ----------
        data : torch_gometric.data.Batch
            Batch graph.

        Return
        ------
        y : torch.tensor
            Per graph predictions of shape `(n_samples, dim)`.
        """
        x = data.x  # (n_nodes_batch, n_node_features)
        edge_index = data.edge_index  # (2, n_edges_batch)
        edge_features = data.edge_features  # (n_edges_batch, n_edge_features=1)
        batch = data.batch  # (n_nodes_batch,)

        x = self.conv(x, edge_index, edge_features)  # (n_nodes_batch, n_channels)

        x = global_mean_pool(x, batch)  # (n_samples, n_channels)

        x = self.fc1(x)  # (n_samples, hidden_size)
        y = self.fc2(x)  # (n_samples, n_targets)

        return y


class TestGradientWRTInput:
    @pytest.mark.parametrize("in_features", [2, 3])
    @pytest.mark.parametrize("out_features", [4, 5])
    def test_linear(self, in_features, out_features):
        torch.manual_seed(34)
        initial_guess = torch.nn.Linear(in_features, out_features, bias=True)
        initial_guess_copy = deepcopy(initial_guess)

        model = Net()
        final_guess = gradient_wrt_input(
            model, initial_guess, Linear.to_graph, n_iter=101
        )
        # Checks
        assert isinstance(final_guess, torch.nn.Linear)

        # No modification in place
        assert torch.allclose(initial_guess.weight, initial_guess_copy.weight)
        assert torch.allclose(initial_guess.bias, initial_guess_copy.bias)

        # Some optimzation done
        assert not torch.allclose(final_guess.weight, initial_guess.weight)
        assert not torch.allclose(final_guess.bias, initial_guess.bias)
