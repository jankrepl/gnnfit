"""Collection of utils for visualization."""
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from gnnfit.convert import MLP


def plot_network_graph(module, to_graph):
    """Plot the simplest graph without making major assumption.

    Parameters
    ----------
    graph : torch.nn.Module
        Input nerual network.

    to_graph : callable
        Convertor that turns `torch.nn.Module` to `torch_geometric.data.Data`.
    """
    network = to_networkx(to_graph(module))
    nx.draw(network, nx.circular_layout(network))
    plt.show()


def plot_MLP(module):
    """Plot multilayer perceptron.

    Parameters
    ----------
    module : torch.nn.Module
        Multilayer perceptron.
    """
    if not MLP._is_mlp(module):
        raise TypeError("The input module is not an MLP")

    # Visualization parameters
    node_distance = (0, -1)  # Distance w.r.t. other nodes
    layer_distance = (5, 0)  # Distance w.r.t the previous layer
    bias_distance = (0.5, 0.5)  # Distance w.r.t. the corresponding node

    node_size = 250  # just a  multiplier to help with adjusting
    weight_size = 2  # just a multiplier to help with adjusting

    node_color = "green"  # the same for all nodes
    edge_color = "black"  # the same for all edges

    # Conversions
    layers = list(MLP._get_learnable_layers(module))
    graph = MLP.to_graph(module, node_strategy="proportional")
    network = to_networkx(graph)  # Does not export node/ edge features

    # Add node attributes
    node_features = list(graph.x.squeeze().detach().numpy())
    for i, node_feature in enumerate(node_features):
        network.nodes[i]["value"] = node_feature * node_size

    # Add edge attributes
    edge_features = list(graph.edge_features.squeeze().detach().numpy())

    for i, edge_feature in enumerate(edge_features):
        start_ix = graph.edge_index[:, i][0].item()
        end_ix = graph.edge_index[:, i][1].item()
        network.edges[start_ix, end_ix]["weight"] = edge_feature * weight_size

    # Generate positions
    pos = []

    # Input layer
    for i in range(layers[0].in_features):
        pos.append(
            (
                -layer_distance[0] + i * node_distance[0],
                -layer_distance[1] + i * node_distance[1],
            )
        )

    for i_layer, layer in enumerate(layers):
        for i_node in range(layer.out_features):
            pos.append(
                (
                    i_layer * layer_distance[0] + i_node * node_distance[0],
                    (i_layer * layer_distance[1] + i_node * node_distance[1]),
                )
            )

        if layer.bias is not None:
            for i_bias in range(layer.out_features):
                pos.append(
                    (
                        i_layer * layer_distance[0]
                        + i_bias * node_distance[0]
                        + bias_distance[0],
                        (
                            i_layer * layer_distance[1]
                            + i_bias * node_distance[1]
                            + bias_distance[1]
                        ),
                    )
                )

    pos_d = {i: p for i, p in enumerate(pos)}

    # Draw
    nx.draw(
        network,
        pos=pos_d,
        width=list(nx.get_edge_attributes(network, "weight").values()),
        node_size=list(nx.get_node_attributes(network, "value").values()),
        node_color=node_color,
        edge_color=edge_color,
        with_labels=True,
    )

    plt.show()


if __name__ == "__main__":

    module = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=True),
        torch.nn.Linear(3, 10, bias=False),
        torch.nn.Linear(10, 20, bias=True),
        torch.nn.Linear(20, 5, bias=True),
        torch.nn.Linear(5, 40, bias=True),
    )

    # plot_network_graph(module, MLP.to_graph)
    plot_MLP(module)
