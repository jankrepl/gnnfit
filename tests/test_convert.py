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
        "module, out",
        [
            (torch.nn.Sequential(torch.nn.Linear(2, 3)), True),
            (torch.nn.Sequential(), False),
            (torch.nn.Linear(4, 5), False),  # To make things simpler
            (Module1(), True),
            (Module2(), False),
            (Module3(), True),
            (Module4(), True),
            (Module5(), False),
            (Module6(), False),
        ],
    )
    def test_is_mlp(self, module, out):
        assert MLP._is_mlp(module) == out


class TestMLPToGraph:
    def test_incorrect_type(self):
        with pytest.raises(TypeError):
            MLP.to_graph("wrong_type")

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("in_features", [2, 3])
    @pytest.mark.parametrize("out_features", [5, 6])
    def test_same_as_linear(self, in_features, out_features, bias):
        module = torch.nn.Linear(in_features, out_features, bias=bias)
        x_linear = Linear.to_graph(module)
        x_mlp = MLP.to_graph(torch.nn.Sequential(module))

        assert x_linear.num_nodes == x_mlp.num_nodes
        assert torch.allclose(x_linear.x, x_mlp.x)
        assert torch.allclose(x_linear.edge_index, x_mlp.edge_index)
        assert torch.allclose(x_linear.edge_features, x_mlp.edge_features)

    @pytest.mark.parametrize(
        "module",
        [
            torch.nn.Sequential(torch.nn.Linear(2, 4)),
            torch.nn.Sequential(torch.nn.Linear(4, 7, bias=False)),
            torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Linear(7, 8)),
            torch.nn.Sequential(
                torch.nn.Linear(4, 7, bias=False),
                torch.nn.Linear(7, 3),
                torch.nn.Linear(3, 2),
            ),
        ],
    )
    def test_n_edges(self, module):
        """Number of edges is equal to the number of parameteres."""
        n_edges_expected = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        graph = MLP.to_graph(module)

        assert len(graph.edge_features) == n_edges_expected
        assert graph.edge_index.shape[1] == n_edges_expected

    def test_biases_correct(self):
        """Bias has not incoming edgese and has exactly one outcoming."""
        module = torch.nn.Sequential(
            torch.nn.Linear(2, 3, bias=True),
            torch.nn.Linear(3, 4, bias=True),
            torch.nn.Linear(4, 2, bias=True),
        )

        is_bias = [
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
        ]
        bias_ids = {i for i, b in enumerate(is_bias) if b}

        graph = MLP.to_graph(module)
        start_nodes = graph.edge_index[0, :].detach().numpy()
        end_nodes = graph.edge_index[1, :].detach().numpy()

        # There are no incoming edges to bias nodes
        assert not (bias_ids & set(end_nodes))

        # Assert the bias nodes are outcoming for exactly 1 edge
        for bias_id in bias_ids:
            assert len([x for x in list(start_nodes) if x == bias_id])

    @pytest.mark.parametrize("random_state", [2, 3])
    @pytest.mark.parametrize("node_strategy", [None, "constant", "proportional"])
    def test_basic_no_bias(self, node_strategy, random_state):
        in_features = 3
        hidden_features = 4
        out_features = 2

        target = torch.tensor([[3.43]])

        torch.manual_seed(random_state)

        linear_1 = torch.nn.Linear(
            in_features=in_features, out_features=hidden_features, bias=False
        )
        linear_2 = torch.nn.Linear(
            in_features=hidden_features, out_features=out_features, bias=False
        )
        module = torch.nn.Sequential(linear_1, linear_2)
        graph = MLP.to_graph(module, target=target, node_strategy=node_strategy)

        # Checks
        if node_strategy is None:
            x_true = None
        elif node_strategy == "constant":
            x_true = torch.ones((9, 1), dtype=torch.float)
        elif node_strategy == "proportional":
            x_true = torch.tensor(
                [
                    1,
                    1,
                    1,
                    1 / 4,
                    1 / 4,
                    1 / 4,
                    1 / 4,
                    1 / 5,
                    1 / 5,
                ]
            )[:, None]

        edge_index_true = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 7, 8, 7, 8, 7, 8],
            ],
            dtype=torch.int64,
        )

        w_1 = linear_1.weight
        w_2 = linear_2.weight
        edge_features_true = torch.tensor(
            [
                [w_1[0, 0]],
                [w_1[1, 0]],
                [w_1[2, 0]],
                [w_1[3, 0]],
                [w_1[0, 1]],
                [w_1[1, 1]],
                [w_1[2, 1]],
                [w_1[3, 1]],
                [w_1[0, 2]],
                [w_1[1, 2]],
                [w_1[2, 2]],
                [w_1[3, 2]],
                [w_2[0, 0]],
                [w_2[1, 0]],
                [w_2[0, 1]],
                [w_2[1, 1]],
                [w_2[0, 2]],
                [w_2[1, 2]],
                [w_2[0, 3]],
                [w_2[1, 3]],
            ]
        )
        assert isinstance(graph, Data)

        assert graph.num_nodes == 9
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
        hidden_features = 4
        out_features = 2

        target = torch.tensor([[3.43]])

        torch.manual_seed(random_state)

        linear_1 = torch.nn.Linear(
            in_features=in_features, out_features=hidden_features, bias=True
        )
        linear_2 = torch.nn.Linear(
            in_features=hidden_features, out_features=out_features, bias=True
        )
        module = torch.nn.Sequential(linear_1, linear_2)
        graph = MLP.to_graph(module, target=target, node_strategy=node_strategy)

        # Checks
        if node_strategy is None:
            x_true = None
        elif node_strategy == "constant":
            x_true = torch.ones((15, 1), dtype=torch.float)
        elif node_strategy == "proportional":
            x_true = torch.tensor(
                [
                    1,
                    1,
                    1,
                    1 / 4,
                    1 / 4,
                    1 / 4,
                    1 / 4,
                    1,
                    1,
                    1,
                    1,
                    1 / 5,
                    1 / 5,
                    1,
                    1,
                ]
            )[:, None]

        edge_index_true = torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    7,
                    8,
                    9,
                    10,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    13,
                    14,
                ],
                [
                    3,
                    4,
                    5,
                    6,
                    3,
                    4,
                    5,
                    6,
                    3,
                    4,
                    5,
                    6,
                    3,
                    4,
                    5,
                    6,
                    11,
                    12,
                    11,
                    12,
                    11,
                    12,
                    11,
                    12,
                    11,
                    12,
                ],
            ],
            dtype=torch.int64,
        )

        w_1 = linear_1.weight
        b_1 = linear_1.bias
        w_2 = linear_2.weight
        b_2 = linear_2.bias
        edge_features_true = torch.tensor(
            [
                [w_1[0, 0]],
                [w_1[1, 0]],
                [w_1[2, 0]],
                [w_1[3, 0]],
                [w_1[0, 1]],
                [w_1[1, 1]],
                [w_1[2, 1]],
                [w_1[3, 1]],
                [w_1[0, 2]],
                [w_1[1, 2]],
                [w_1[2, 2]],
                [w_1[3, 2]],
                [b_1[0]],
                [b_1[1]],
                [b_1[2]],
                [b_1[3]],
                [w_2[0, 0]],
                [w_2[1, 0]],
                [w_2[0, 1]],
                [w_2[1, 1]],
                [w_2[0, 2]],
                [w_2[1, 2]],
                [w_2[0, 3]],
                [w_2[1, 3]],
                [b_2[0]],
                [b_2[1]],
            ]
        )
        assert isinstance(graph, Data)

        assert graph.num_nodes == 15
        assert torch.equal(graph.y, target)
        assert torch.equal(graph.edge_index, edge_index_true)
        assert torch.equal(graph.edge_features, edge_features_true)
        if x_true is not None:
            assert torch.equal(graph.x, x_true)
        else:
            assert graph.x is None
