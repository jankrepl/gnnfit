"""Tests focused on th `convert` module."""
import pytest
import torch
from torch_geometric.data import Data

from gnnfit.convert import Linear


class TestLinearToData:
    def test_incorrect_type(self):
        with pytest.raises(TypeError):
            Linear.to_data("wrong_type")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            Linear.to_data(torch.nn.Linear(4, 5), node_strategy="nonexistent")

    @pytest.mark.parametrize("random_state", [2, 3])
    @pytest.mark.parametrize("node_strategy", [None, "constant", "proportional"])
    def test_basic_no_bias(self, node_strategy, random_state):
        in_features = 3
        out_features = 2
        target = torch.tensor([[3.43]])

        torch.manual_seed(random_state)
        model = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

        data = Linear.to_data(model, target=target, node_strategy=node_strategy)

        # Checks
        if node_strategy is None:
            x_true = None
        elif node_strategy == "constant":
            x_true = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)
        elif node_strategy == "proportional":
            x_true = torch.tensor([1, 1, 1, 1 / 4, 1 / 4])

        edge_index_true = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [3, 4, 3, 4, 3, 4]], dtype=torch.int64
        )

        w = model.weight
        edge_features_true = torch.tensor(
            [
                [w[0, 0]],
                [w[1, 0]],
                [w[0, 1]],
                [w[1, 1]],
                [w[0, 2]],
                [w[1, 2]],
            ]
        )
        assert isinstance(data, Data)

        assert data.num_nodes == 5
        assert torch.equal(data.y, target)
        assert torch.equal(data.edge_index, edge_index_true)
        assert torch.equal(data.edge_features, edge_features_true)
        if x_true is not None:
            assert torch.equal(data.x, x_true)
        else:
            assert data.x is None

    @pytest.mark.parametrize("random_state", [2, 3])
    @pytest.mark.parametrize("node_strategy", [None, "constant", "proportional"])
    def test_basic_with_bias(self, node_strategy, random_state):
        in_features = 3
        out_features = 2
        target = torch.tensor([[3.43]])

        torch.manual_seed(random_state)
        model = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )

        data = Linear.to_data(model, target=target, node_strategy=node_strategy)

        # Checks
        if node_strategy is None:
            x_true = None
        elif node_strategy == "constant":
            x_true = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float)
        elif node_strategy == "proportional":
            x_true = torch.tensor([1, 1, 1, 1 / 4, 1 / 4, 1, 1])

        edge_index_true = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 5, 6], [3, 4, 3, 4, 3, 4, 3, 4]], dtype=torch.int64
        )

        w = model.weight
        b = model.bias
        edge_features_true = torch.tensor(
            [
                [w[0, 0]],
                [w[1, 0]],
                [w[0, 1]],
                [w[1, 1]],
                [w[0, 2]],
                [w[1, 2]],
                [b[0]],
                [b[1]],
            ]
        )
        assert isinstance(data, Data)

        assert data.num_nodes == 7
        assert torch.equal(data.y, target)
        assert torch.equal(data.edge_index, edge_index_true)
        assert torch.equal(data.edge_features, edge_features_true)
        if x_true is not None:
            assert torch.equal(data.x, x_true)
        else:
            assert data.x is None


class TestToModule:
    @pytest.mark.parametrize("random_state", [1, 3])
    def test_without_bias(self, random_state):
        torch.manual_seed(random_state)
        # 2 * 4 | in_features x out_features

        w = torch.randn(4, 2)
        data = Data(
            edge_index=torch.tensor(
                [[0, 0, 0, 0, 1, 1, 1, 1], [2, 3, 4, 5, 2, 3, 4, 5]]
            ),
            edge_features=torch.tensor(
                [
                    [w[0, 0]],
                    [w[1, 0]],
                    [w[2, 0]],
                    [w[3, 0]],
                    [w[0, 1]],
                    [w[1, 1]],
                    [w[2, 1]],
                    [w[3, 1]],
                ]
            ),
        )
        model = Linear.to_module(data)

        model_true = torch.nn.Linear(2, 4, bias=False)
        model_true.weight = torch.nn.Parameter(w)

        assert isinstance(model_true, torch.nn.Linear)
        assert torch.equal(model_true.weight, model.weight)
        assert model.bias is None

    @pytest.mark.parametrize("random_state", [1, 3])
    def test_with_bias(self, random_state):
        torch.manual_seed(random_state)
        # 2 * 4 | in_features x out_features

        w = torch.randn(4, 2)
        b = torch.randn(4)

        data = Data(
            edge_index=torch.tensor(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 6, 7, 8, 9],
                    [2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
                ]
            ),
            edge_features=torch.tensor(
                [
                    [w[0, 0]],
                    [w[1, 0]],
                    [w[2, 0]],
                    [w[3, 0]],
                    [w[0, 1]],
                    [w[1, 1]],
                    [w[2, 1]],
                    [w[3, 1]],
                    [b[0]],
                    [b[1]],
                    [b[2]],
                    [b[3]],
                ]
            ),
        )
        model = Linear.to_module(data)
        model_true = torch.nn.Linear(2, 4, bias=True)
        model_true.weight = torch.nn.Parameter(w)
        model_true.bias = torch.nn.Parameter(b)

        assert isinstance(model_true, torch.nn.Linear)
        assert torch.equal(model_true.weight, w)
        assert torch.equal(model.bias, b)

    @pytest.mark.parametrize("in_features", [2, 3])
    @pytest.mark.parametrize("out_features", [4, 8])
    @pytest.mark.parametrize("bias", [True, False], ids=["with_bias", "no_bias"])
    @pytest.mark.parametrize("random_state", [5, 7])
    def test_round_trip(self, in_features, out_features, bias, random_state):
        torch.manual_seed(random_state)
        input_model = torch.nn.Linear(in_features, out_features, bias=bias)

        output_model = Linear.to_module(Linear.to_data(input_model))

        assert torch.equal(input_model.weight, output_model.weight)
        if bias:
            assert torch.equal(input_model.bias, output_model.bias)
