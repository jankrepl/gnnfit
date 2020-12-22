"""Conversion utilities."""
from abc import ABC, abstractmethod
from collections import Counter
from itertools import product
from unittest.mock import Mock

import torch
from torch_geometric.data import Data


class Convertor(ABC):
    """Template for conversions between torch and torch_geometric objects."""

    @staticmethod
    @abstractmethod
    def to_graph():
        """Convert from `torch.nn.Linear` to `torch_geometric.data.Data`."""

    @staticmethod
    @abstractmethod
    def to_module():
        """Convert from `torch_geometric.data.Data` to `torch.nn.Module`."""


class Linear(Convertor):
    """Conversions between `torch.nn.Linear` and `torch_geometric.data.Data`."""

    @staticmethod
    def to_graph(module, target=None, node_strategy="proportional"):
        """Convert `torch.nn.Linear` to `torch_geometric.data.Data`.

        Parameters
        ----------
        module : torch.nn.Linear
            Linear module that might contain a bias.

        target : None or torch.Tensor
            If specified, than represents a supervised target of the module. The
            expected shape is `(1, target_dim)`.

        node_strategy : None or str, {"constant", "proportional"}
            If not specified, then then no node features used. If specified,
            it can be the following options:

                - "constant" : all node features are equal to 1
                - "proportional" : the node feature of a given node
                  is equal to `1 / (n_parent_nodes + 1)`.

        Returns
        -------
        graph : torch_geometric.data.Data
            Graph that is ready to be used with `torch_geometric`.
        """
        if not isinstance(module, torch.nn.Linear):
            raise TypeError("The module needs to be a torch.nn.Linear instance.")

        if module.bias is not None:
            n_bias_nodes = module.out_features
        else:
            n_bias_nodes = 0

        n_nodes = module.in_features + module.out_features + n_bias_nodes

        # Edge index
        start_inp = 0
        end_inp = start_inp + module.in_features  # noninclusive
        start_out = end_inp
        end_out = start_out + module.out_features  # noninclusive

        edge_index_l_weights = [
            x for x in product(range(start_inp, end_inp), range(start_out, end_out))
        ]
        edge_index_l_bias = [(end_out + i, start_out + i) for i in range(n_bias_nodes)]
        edge_index = (
            torch.tensor(edge_index_l_weights + edge_index_l_bias, dtype=torch.int64)
            .t()
            .contiguous()
        )

        # Edge features
        edge_features = torch.cat(
            [
                module.weight.t().flatten(),
                torch.empty(0) if module.bias is None else module.bias,
            ]
        )[:, None]

        # Node features
        if node_strategy is None:
            x = None
        elif node_strategy == "constant":
            x = torch.ones(n_nodes, dtype=torch.float)
            x = x[:, None]
        elif node_strategy == "proportional":
            x = torch.ones(n_nodes, dtype=torch.float)
            x[start_out:end_out] = 1 / (1 + module.in_features)
            x = x[:, None]

        else:
            raise ValueError(f"Unsupported node strategy {node_strategy}")

        graph = Data(x=x, edge_index=edge_index, edge_features=edge_features, y=target)

        graph.num_nodes = n_nodes

        return graph

    @staticmethod
    def to_module(graph):
        """Convert `torch_geometric.data.Data` to `torch.nn.Linear`.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            Graph that conpatible with `torch_geometric`.

        Returns
        -------
        module : torch.nn.Linear
            Linear module that might contain bias.
        """
        edge_features = graph.edge_features

        output_indices = set(graph.edge_index[1, :].numpy())
        counter = Counter(graph.edge_index[0, :].numpy())
        bias_indices = {i for i, count in counter.items() if count == 1}
        input_indices = {i for i, count in counter.items() if count > 1}

        in_features = len(input_indices)
        out_features = len(output_indices)

        module = torch.nn.Linear(in_features, out_features, bias=bool(bias_indices))

        weight = torch.nn.Parameter(
            edge_features[: in_features * out_features]
            .view(in_features, out_features)
            .t()
        )
        module.weight = weight

        if bias_indices:
            module.bias = torch.nn.Parameter(
                edge_features[in_features * out_features :].squeeze()
            )

        return module


class MLP(Convertor):
    """Conversion between a Multi-layer Perceptron and `torch_geometric.data.Data`.

    Note that we define a MLP as a subclass of `torch.nn.Module` with all
    learnable layers being `torch.nn.Linear`. This for example means,
    that activation layers (without learnable parameters) are allowed.
    """

    @staticmethod
    def to_graph(module, target=None, node_strategy="proportional"):
        """Convert MLP `torch.nn.Module` to `torch_geometric.data.Data`.

        Some principles for the algorithm:

            1) We go from left to right when creating the graph (=forward pass)
               in batches of 2 - this can be seen as linear layers
            2) Bias nodes are only added for output layers.

        Parameters
        ----------
        module : torch.nn.Module
            MLP module.

        target : None or torch.Tensor
            If specified, than represents a supervised target of the module. The
            expected shape is `(1, target_dim)`.

        node_strategy : None or str, {"constant", "proportional"}
            If not specified, then then no node features used. If specified,
            it can be the following options:

                - "constant" : all node features are equal to 1
                - "proportional" : the node feature of a given node
                  is equal to `1 / (n_parent_nodes + 1)`.

        Returns
        -------
        graph : torch_geometric.data.Data
            Graph that is ready to be used with `torch_geometric`.
        """
        if not MLP._is_mlp(module):
            raise TypeError("The provided module is not an MLP")

        layers = list(MLP._get_learnable_layers(module))

        first_linear = Mock(spec=torch.nn.Linear)
        first_linear.in_features = 0
        first_linear.out_features = layers[0].in_features
        first_linear.bias = None
        first_linear.weight = torch.empty(first_linear.out_features, 0)

        layers = [first_linear, *layers]

        # Prepare input layer
        x_l = []
        edge_index_l = []
        edge_features_l = []

        start_inp = 0
        start_out = first_linear.in_features

        n_nodes = 0

        for i, linear in enumerate(layers):
            if linear.bias is not None:
                n_bias_nodes = linear.out_features
            else:
                n_bias_nodes = 0

            start_bias = start_out + linear.out_features

            n_new_nodes = linear.out_features + n_bias_nodes
            n_nodes += n_new_nodes

            # Node features
            if node_strategy is None:
                x_l = None
            elif node_strategy == "constant":
                x_l.append(torch.ones((n_new_nodes, 1), dtype=torch.float))
            elif node_strategy == "proportional":
                x_ = torch.ones(n_new_nodes, dtype=torch.float)
                x_[: linear.out_features] = 1 / (1 + linear.in_features)
                x_l.append(x_[:, None])
            else:
                raise ValueError(f"Unsupported node strategy {node_strategy}")

            # Edge features
            edge_features_ = torch.cat(
                [
                    linear.weight.t().flatten(),
                    torch.empty(0) if linear.bias is None else linear.bias,
                ]
            )[:, None]
            edge_features_l.append(edge_features_)

            # Edge index
            edge_index_l_weights = [
                x
                for x in product(
                    range(start_inp, start_inp + linear.in_features),
                    range(start_out, start_out + linear.out_features),
                )
            ]
            edge_index_l_bias = [
                (start_bias + i, start_out + i) for i in range(n_bias_nodes)
            ]
            edge_index_ = (
                torch.tensor(
                    edge_index_l_weights + edge_index_l_bias, dtype=torch.int64
                )
                .t()
                .contiguous()
            )
            edge_index_l.append(edge_index_)

            # Preparation for the next layer
            start_inp += linear.in_features
            start_out += linear.out_features + n_bias_nodes

        x = torch.cat(x_l) if x_l is not None else None
        edge_index = torch.cat(edge_index_l, dim=-1)
        edge_features = torch.cat(edge_features_l)

        graph = Data(x=x, edge_index=edge_index, edge_features=edge_features, y=target)
        graph.num_nodes = n_nodes

        return graph

    @staticmethod
    def to_module(graph):
        """Convert `torch_geometric.data.Data` to `torch.nn.Linear`.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            Graph that conpatible with `torch_geometric`.

        Returns
        -------
        module : torch.nn.Linear
            Linear module that might contain bias.
        """
        raise NotImplementedError

    @staticmethod
    def _is_mlp(module):
        """Check whether an MLP.

        The condition is that all layers with learnable parameters
        need to be `torch.nn.Linear`. It is not allowed to have
        an empty model.

        Parameters
        ----------
        module : torch.nn.Module
            Any module.

        Returns
        -------
        bool
            If true, then the module is a MLP.
        """
        if not isinstance(module, torch.nn.Module):
            return False

        has_learnable_layers = False
        for layer in MLP._get_learnable_layers(module):
            has_learnable_layers = True
            if not isinstance(layer, torch.nn.Linear):
                return False

        return has_learnable_layers

    @staticmethod
    def _get_learnable_layers(module):
        """Extract only learnable layers from a `torch.nn.Module`.

        The original order is left untouched. And no recursion is taking place
        (only 1st level hieararchy iteration).

        Parameters
        ----------
        module : torch.nn.Module
            Any module.

        Yields
        ------
        learnable_layer : torch.nn.Module
            A learnable layer.
        """
        for layer in module.modules():
            # skip itself
            if layer is module:
                continue

            try:
                next(layer.parameters())
                yield layer

            except StopIteration:
                continue
