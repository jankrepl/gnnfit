"""Optimization utilities."""
from copy import deepcopy

import torch
from torch_geometric.data import Batch


def gradient_wrt_input(
    model,
    initial_guess,
    to_graph,
    n_iter=1000,
    lr=1e-1,
    verbose=True,
    device=None,
    dtype=None,
):
    """Compute gradient with respect to the edge features of the input graph.

    In other words, given a (trained) GNN we would like to find
    the optimal weights of the inner `torch.nn.Module` minimizing
    the loss.

    Parameters
    ----------
    model : torch.nn.Module
        The Graph Neural Network. We assume that the network has a single
        regression output that we are trying to minimize.

    initial_guess : torch.nn.Module
        Initial guess on the inner network.

    to_graph : callable
        Convertor for the `initial_guess` neural network to a graph.

    n_iter : int
        Number of iterations.

    lr : float
        Learning rate.

    verbose : bool
        If True, info massages will be printed.

    device : torch.device or None
        Device to use. If None then `torch.device("cpu")` is used.

    dtype : torch.dtype or None
        Dtype to use. If None then `torch.float32` is used.

    Returns
    -------
    final_guess : torch.nn.Module
        Inner network that minimizes the loss of the `model`.

    history : list
        List of floats representing per iteration loss.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    final_guess = deepcopy(initial_guess)

    for param in final_guess.parameters():
        param.requires_grad = True

    model = model.to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.Adam(final_guess.parameters(), lr=lr)

    history = []

    if verbose:
        print("Starting optimization")

    for i in range(n_iter):
        if i % 100 == 0 and verbose and i != 0:
            msg = f"{i}-th iteration, loss: {history[-1]:.4f}"
            print(msg)

        # Convert to Batch graph
        graph = to_graph(final_guess)
        batch_graph = Batch.from_data_list([graph])

        # GNN forward pass
        loss = model(batch_graph)

        # Store result
        history.append(loss.item())

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        print(f"Optimization done, final loss: {history[-1]:.4f}")

    return final_guess, history
