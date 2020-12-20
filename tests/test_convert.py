"""Tests focused on th `convert` module."""
import pytest
import torch
from torch_geometric.data import Data

from gnnfit.convert import Linear, MLP


class TestLinearToGraph:
    def test_incorrect_type(self):
        with pytest.raises(TypeError):
            Linear.to_graph("wrong_type")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            Linear.to_graph(torch.nn.Linear(4, 5), node_strategy="nonexistent")

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

        graph = Linear.to_graph(model, target=target, node_strategy=node_strategy)

        # Checks
        if node_strategy is None:
            x_true = None
        elif node_strategy == "constant":
            x_true = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)[:, None]
        elif node_strategy == "proportional":
            x_true = torch.tensor([1, 1, 1, 1 / 4, 1 / 4])[:, None]

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
        assert isinstance(graph, Data)

        assert graph.num_nodes == 5
        assert torch.equal(graph.y, target)
        assert torch.equal(graph.edge_index, edge_index_true)
        assert torch.equal(graph.edge_features, edge_features_true)
        if x_true is not None:
            assert torch.equal(graph.x, x_true)
        else:
            assert graph.x is None

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

        graph = Linear.to_graph(model, target=target, node_strategy=node_strategy)

        # Checks
        if node_strategy is None:
            x_true = None
        elif node_strategy == "constant":
            x_true = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float)[:, None]
        elif node_strategy == "proportional":
            x_true = torch.tensor([1, 1, 1, 1 / 4, 1 / 4, 1, 1])[:, None]

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
        assert isinstance(graph, Data)

        assert graph.num_nodes == 7
        assert torch.equal(graph.y, target)
        assert torch.equal(graph.edge_index, edge_index_true)
        assert torch.equal(graph.edge_features, edge_features_true)
        if x_true is not None:
            assert torch.equal(graph.x, x_true)
        else:
            assert graph.x is None


class TestToModule:
    @pytest.mark.parametrize("random_state", [1, 3])
    def test_without_bias(self, random_state):
        torch.manual_seed(random_state)
        # 2 * 4 | in_features x out_features

        w = torch.randn(4, 2)
        graph = Data(
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
        model = Linear.to_module(graph)

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

        graph = Data(
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
        model = Linear.to_module(graph)
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

        output_model = Linear.to_module(Linear.to_graph(input_model))

        assert torch.equal(input_model.weight, output_model.weight)
        if bias:
            assert torch.equal(input_model.bias, output_model.bias)


class TestMLPUtils:
    class Module1(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = torch.nn.Linear(4, 5)

    class Module2(torch.nn.Module):
        pass

    class Module3(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = torch.nn.Linear(4, 5)
            self.fc2 = torch.nn.Linear(5, 10)

    class Module4(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = torch.nn.Linear(4, 5)
            self.fc2 = torch.nn.Linear(5, 10)
            self.act = torch.nn.ReLU()

    class Module5(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = torch.nn.Linear(4, 5)
            self.conv1 = torch.nn.Conv1d(5, 2, 5)

    class Module6(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer = torch.nn.Sequential(torch.nn.Linear(3, 4))

    @pytest.mark.parametrize(
        "module, out", [
                        (torch.nn.Sequential(torch.nn.Linear(2, 3)), True),
                        (torch.nn.Sequential(), False),
                        (Module1(), True),
                        (Module2(), False),
                        (Module3(), True),
                        (Module4(), True),
                        (Module5(), False),
                        (Module6(), False),
                        ]
)
    def test_check_mlp(self, module, out):
        assert MLP._check_mlp(module) == out
