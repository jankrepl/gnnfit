"""Data creation utilites."""

from copy import deepcopy
import numpy as np

import torch.optim
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from tqdm import tqdm


class InnerDataset(Dataset):
    """The dataset that will be used to generate the `OuterDataset`.

    Parameters
    ----------
    X : np.ndarray
        Features array of shape `(n_samples, m_1, ..., m_k)`.

    y : np.ndarray
        Targets array of shape `(n_samples, n_1, ..., n_k)`.

    """

    def __init__(self, X, y, transform=None):
        if len(X) != len(y):
            raise ValueError("The features and targets have differnt number of samples")

        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        """Compute length of the dataset."""
        return len(self.y)

    def __getitem__(self, i):
        """Get a single sample."""
        X_sample, y_sample = self.X[i], self.y[i]

        if self.transform:
            X_sample, y_sample = self.transform(X_sample, y_sample)

        return X_sample, y_sample


class OuterDatasetCreator:
    """Utility class for creating dataset via training of the underlying model.

    Parameters
    ----------
    model_factory : callable
        Function that returns a `torch.nn.Module`. Keyword arguments can
        be passed dynamically for each training.

    dataset : InnerDataset
        Dataset that yields pars of X and y.

    loss : callable
        Function that inputs `y_true` and `y_pred` and returns a number. Note
        that the inputs are going to be `torch.Tensor`, however, one
        does not have to worry about breaking the computational graph.

    custom_losses : None or dict
        If specified, the keys are names and the values are callables
        representing custom losses.

    batch_size_losses : int
        Batch size to be used for computation of train (and validation)
        losses. Only forward pass is required.

    train_indices : None or array-like
        If specified, then a 1D array represnting the indices of the dataset
        to be used for training. If not specified, then all of them
        are used.

    val_indices : None or array-like
        If specified, then a 1D array representing the indices of the dataset
        to be used for validation. If not specified, the complement of
        the `train_indices` is used. In case there are validation indices, then
        all validation losses are populated with `None`.
    """

    def __init__(
        self,
        model_factory,
        dataset,
        loss,
        custom_losses=None,
        batch_size_losses=32,
        train_indices=None,
        val_indices=None,
        optimizer=None,
    ):
        self.model_factory = model_factory
        self.dataset = dataset
        self.loss = loss
        self.custom_losses = custom_losses or {"main": loss}
        self.batch_size_losses = batch_size_losses
        self.train_indices = (
            np.arange(len(dataset)) if train_indices is None else train_indices
        )

        if val_indices is None:
            all_indices = np.arange(len(dataset))
            self.val_indices = np.sort(np.setdiff1d(all_indices, self.train_indices))

        else:
            self.val_indices = val_indices

        # Checks
        if len(np.intersect1d(self.train_indices, self.val_indices)) > 0:
            raise ValueError("The validation and training sets cannot overlap.")

    def generate(
        self,
        n_trainings=1,
        n_epochs=1,
        batch_size=32,
        save_frequency=1,
        model_factory_kwargs=None,
        optimizer_class="Adam",
        optimizer_kwargs=None,
        dtype=torch.float32,
        device="cpu",
        random_state=None,
    ):
        """Generata data.

        Parameters
        ----------
        n_trainings : int
            Number of separate trainings to be done. The model used
            for each training will be deterimed via `model_factory` function.
        n_epochs : int
            Number of epochs for each of the trainings.

        batch_size : int
            Batch size used for training.

        save_frequency : int
            Determines how often we save the datapoint. If `save_frequency==1`,
            then we save after each gradient step. If `save_frequency==5` we
            save every 5th gradient step. In other words, the lower this number,
            the higher the number of collected samples.

        model_factory_kwargs : None or dict or list
            Additionally parameters to be passed into the `model_factory`. If
            None, nothing is passed. If `dict` then the same parameters
            are used for all trainings. If `list` then the elements are of type
            `dict` and they represent per training specifc model kwargs.

        optimizer_class : str
            Name of the optimization class inside of `torch.optim`.

        optimizer_kwargs : str
            The keyword arguments passed to the optimizer, i.e. the learning
            rate.

        dtype : torch.dtype
            Dtype of the inner dataset batches will be casted to together with
            the the model.

        device : torch.device or str
            Device the batches of the inner dataset will be put to together
            with the model.
        """
        model_factory_kwargs_actual = self._handle_kwargs(
            n_trainings, model_factory_kwargs
        )

        i_batch = 0

        for i_train, kwargs in enumerate(model_factory_kwargs_actual):
            model = self.model_factory(**kwargs).train()
            model = model.to(device=device, dtype=dtype)
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                drop_last=True,  # make sure not crazy bumps
                sampler=SubsetRandomSampler(self.train_indices),
            )

            optimizer = getattr(torch.optim, optimizer_class)(
                model.parameters(), **(optimizer_kwargs or {})
            )

            iterable = tqdm(range(n_epochs), desc=f"Traninig #{i_train + 1}")
            for i_epoch in iterable:
                for X_batch, y_batch in dataloader:
                    model.train()
                    # Compute gradient step
                    X_batch = X_batch.to(device=device, dtype=dtype)
                    y_batch = y_batch.to(device=device, dtype=dtype)

                    optimizer.zero_grad()
                    y_pred_batch = model(X_batch)
                    loss_batch = self.loss(y_batch, y_pred_batch)
                    loss = loss_batch.mean()
                    loss.backward()
                    optimizer.step()

                    iterable.set_postfix({"batch_loss": loss.item()})

                    # Compute losses on the entire sets
                    if i_batch % save_frequency == 0:
                        losses_values = {}
                        model.eval()

                        with torch.no_grad():
                            for loss_name, loss_cal in self.custom_losses.items():
                                for dataset_type, indices in [
                                    ("train", self.train_indices),
                                    ("val", self.val_indices),
                                ]:

                                    entry_name = f"{dataset_type}_{loss_name}"
                                    loss_val = self.compute_loss(
                                        loss_cal,
                                        model,
                                        self.dataset,
                                        indices,
                                        batch_size=self.batch_size_losses,
                                        dtype=dtype,
                                        device=device,
                                    )
                                    losses_values[entry_name] = loss_val

                        yield (deepcopy(model), losses_values)
                    i_batch += 1

    @staticmethod
    def _handle_kwargs(n_trainings, model_factory_kwargs):
        """Handle and check model factory kwargs."""
        if model_factory_kwargs is None:
            model_factory_kwargs_actual = n_trainings * [{}]

        elif isinstance(model_factory_kwargs, dict):
            model_factory_kwargs_actual = n_trainings * [model_factory_kwargs]

        elif isinstance(model_factory_kwargs, list):
            if len(model_factory_kwargs) != n_trainings:
                raise ValueError(
                    "The number of kwargs and the number of trainings is different"
                )
            model_factory_kwargs_actual = model_factory_kwargs

        else:
            raise TypeError(f"Unsupported type {type(model_factory_kwargs)}")

        return model_factory_kwargs_actual

    @staticmethod
    def compute_loss(
        loss,
        model,
        dataset,
        indices=None,
        batch_size=32,
        dtype=torch.float32,
        device="cpu",
    ):
        """Compute a loss over an entire dataset (not just a batch).

        Parameters
        ----------
        loss : callable
            It has two arguments `y_true` and `y_pred` that are tensors
            of the same length. The first dimension is the sample dimension.
            It returns a per sample loss of shape `(n_samples,)`. Importatly,
            one can break out of the torch's computation graph since
            this loss is not going to be backpropagated.

        model : torch.nn.Module
            A network.

        dataset : InnerDataset
            Instance of an `InnerDataset`.

        indices : None or array-like
            If specified, then represents the indices of `dataset` to compute
            the loss over. If not specified, then all indices are used.

        batch_size : int
            Batches over which the loss is computed. Does not have an
            effect on the final average loss over the dataset.

        dtype : torch.dtype
            Dtype of the inner dataset batches will be casted to together with
            the the model.

        device : torch.device or str
            Device the batches of the inner dataset will be put to together
            with the model.


        Returns
        -------
        average_loss : float
            Average loss over the `dataset[indices]`.
        """
        if indices is None:
            indices = np.arange(len(dataset))  # pragma: no cover

        subset = Subset(dataset, indices)
        if len(subset) == 0:
            return None

        model.eval()
        model = model.to(device=device, dtype=dtype)

        dataloader = DataLoader(subset, batch_size=batch_size, drop_last=False)

        all_losses = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device=device, dtype=dtype)
            y_batch = y_batch.to(device=device, dtype=dtype)

            y_pred_batch = model(X_batch)
            loss_batch = loss(y_batch, y_pred_batch)  # (n_samples,)
            all_losses.append(loss_batch)

        all_losses_t = torch.cat(all_losses)

        return all_losses_t.mean().item()
