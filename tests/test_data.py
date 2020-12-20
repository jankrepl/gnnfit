"""Collection of tests focusing on the `data` module."""
import math

import pytest

import numpy as np
from sklearn.datasets import make_regression
import torch

from gnnfit.data import InnerDataset, OuterDatasetCreator


@pytest.fixture()
def inner_dataset_r():
    X, y = make_regression(90, 5, n_informative=4)

    return InnerDataset(X, y)


class TestInnerDataset:
    def test_error(self):
        with pytest.raises(ValueError):
            InnerDataset(np.ones((2, 4)), np.ones(5))

    def test_overall_without_transform(self):
        X = np.random.random((2, 4))
        y = np.random.random(2)

        dataset = InnerDataset(X, y)

        assert len(dataset) == 2

        X_sample, y_sample = X[1], y[1]
        np.testing.assert_array_equal(X_sample, dataset[1][0])
        np.testing.assert_array_equal(y_sample, dataset[1][1])

    def test_overall_with_transform(self):
        X = np.random.random((2, 4))
        y = np.random.random(2)

        dataset = InnerDataset(X, y, transform=lambda x, y: (-x, -y))

        assert len(dataset) == 2

        X_sample, y_sample = X[1], y[1]
        np.testing.assert_array_equal(X_sample, -dataset[1][0])
        np.testing.assert_array_equal(y_sample, -dataset[1][1])


class TestCreateDataByTraining:
    @staticmethod
    def mse(y_true, y_pred):
        y_true_ = y_true.squeeze()
        y_pred_ = y_pred.squeeze()

        if y_true_.ndim != 1 or y_pred_.ndim != 1:
            raise ValueError

        return (y_true_ - y_pred_) ** 2

    def test_errors(self, inner_dataset_r):
        with pytest.raises(ValueError):
            OuterDatasetCreator(
                lambda: torch.nn.Linear(3, 4),
                inner_dataset_r,
                lambda y_true, y_pred: (y_true - y_pred) ** 2,
                train_indices=np.array([1, 2]),
                val_indices=np.array([2, 3]),
            )

    @pytest.mark.parametrize(
        "n_trainings, kwargs, output_or_error",
        (
            (3, None, [{}, {}, {}]),
            (2, {"a": 3}, [{"a": 3}, {"a": 3}]),
            (2, [{"b": 4}, {"c": 5}], [{"b": 4}, {"c": 5}]),
            (5, [{}, {}], ValueError),
            (23, "wrong_type", TypeError),
        ),
    )
    def test_hande_kwargs(self, n_trainings, kwargs, output_or_error):
        if output_or_error in [ValueError, TypeError]:
            with pytest.raises(output_or_error):
                OuterDatasetCreator._handle_kwargs(n_trainings, kwargs)
        else:
            output_actual = OuterDatasetCreator._handle_kwargs(n_trainings, kwargs)
            assert output_actual == output_or_error

    @pytest.mark.parametrize("n_trainings", [1, 3])
    @pytest.mark.parametrize("n_epochs", [2, 4], ids=["n_epochs_2", "n_epochs_4"])
    @pytest.mark.parametrize("batch_size", [5, 6], ids=["batch_size_5", "batch_size_6"])
    @pytest.mark.parametrize("saving_frequency", [1, 7], ids=["freq_1", "freq_7"])
    def test_overall(
        self, inner_dataset_r, n_trainings, n_epochs, batch_size, saving_frequency
    ):
        n_batches_per_dataset = int(len(inner_dataset_r) // batch_size)
        n_batches_overall = n_trainings * n_epochs * n_batches_per_dataset

        n_features = inner_dataset_r[0][0].shape[0]
        odc = OuterDatasetCreator(
            lambda: torch.nn.Linear(n_features, 1), inner_dataset_r, self.mse
        )

        # Check after construction state
        assert len(odc.train_indices) == len(inner_dataset_r)
        assert len(odc.val_indices) == 0  # complement done correctly

        # generate
        out = list(
            odc.generate(
                n_trainings=n_trainings,
                n_epochs=n_epochs,
                batch_size=batch_size,
                save_frequency=saving_frequency,
            )
        )

        assert all(isinstance(x, tuple) for x in out)
        assert all(isinstance(x[0], torch.nn.Module) for x in out)
        assert all(isinstance(x[1], dict) for x in out)

        expected_len = math.ceil(n_batches_overall / saving_frequency)

        assert len(out) == expected_len
