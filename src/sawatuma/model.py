from typing import Tuple, Union
from sawatuma.datasets import ListeningCountsDataset
import numpy as np
import scipy
from scipy import sparse
from tqdm import tqdm


def _dataset_to_matricies(
    dataset: ListeningCountsDataset,
    confidence_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    preferences = np.zeros(
        (dataset.parameters.user_count(), dataset.parameters.track_count())
    )
    confidences = np.zeros_like(preferences)

    for count in dataset:
        preferences[count.user_id - 1, count.track_id - 1] = 1
        confidences[count.user_id - 1, count.track_id - 1] = (
            count.count * confidence_factor + 1
        )

    return (preferences, confidences)


def _mean_squared_error(true: np.ndarray, predicted: np.ndarray) -> float:
    return np.average((true - predicted) ** 2, axis=0)


class Model:
    def __init__(
        self,
        dataset: ListeningCountsDataset,
        factor_count: int,
        confidence_factor: float,
        regularization_factor: float,
    ) -> None:
        self.data_parameters = dataset.parameters
        print("converting datasets into matricies")
        self.preferences, self.confidences = _dataset_to_matricies(
            dataset, confidence_factor
        )

        self.user_factors = np.random.random(
            (self.data_parameters.user_count(), factor_count)
        )
        self.track_factors = np.random.random(
            (self.data_parameters.track_count(), factor_count)
        )

        self.regularization_factor = regularization_factor
        self.confidence_factor = confidence_factor

    # transcribed from http://yifanhu.net/PUB/cf.pdf alongside http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html
    # variable names assume that `preferences` and `confidences` users as the first dimension and `track_factors` are the factors of the tracks,
    # but this works perfectly fine if the two are swapped
    def _als_step(
        self,
        preferences: np.ndarray,
        confidences: np.ndarray,
        track_factors: np.ndarray,
    ) -> np.ndarray:
        FTF = track_factors.T @ track_factors

        user_count = list(preferences.shape)[0]
        factor_count = list(track_factors.shape)[1]
        user_factors = np.zeros((user_count, factor_count))

        for index, (preferences, confidences) in tqdm(
            enumerate(zip(preferences, confidences)), total=user_count
        ):
            C = sparse.diags(confidences)
            CI = sparse.diags(confidences - 1)
            FTCIF = track_factors.T @ sparse.csr_matrix.dot(CI, track_factors)
            FTCF = FTF + FTCIF
            FTCF_reg = FTCF + self.regularization_factor * np.identity(factor_count)
            FTCF_reg_inv = np.linalg.inv(FTCF_reg)
            user_factors[index] = FTCF_reg_inv @ track_factors.T @ C @ preferences

        return user_factors

    def _step(self):
        print("  optimizing user factors")
        self.user_factors = self._als_step(
            self.preferences, self.confidences, self.track_factors
        )

        print("  optimizing track factors")
        self.track_factors = self._als_step(
            self.preferences.T, self.confidences.T, self.user_factors
        )

    @staticmethod
    def _evaluate(dataset: np.ndarray, predictions: np.ndarray) -> float:
        nonzero = np.nonzero(dataset)
        return _mean_squared_error(dataset[nonzero], predictions[nonzero])

    def predict(self):
        return self.user_factors @ self.track_factors.T

    def train(self, num_epochs: int, test: ListeningCountsDataset | None):
        test_preferences = None
        if test is not None:
            test_preferences, _ = _dataset_to_matricies(test, self.confidence_factor)

        print("initial values")
        predictions = self.predict()
        print(f"  train evaluation: {self._evaluate(self.preferences, predictions)}")
        if test_preferences is not None:
            print(f"  test evaluation: {self._evaluate(test_preferences, predictions)}")

        for epoch in range(num_epochs):
            print(f"epoch number {epoch + 1}")
            self._step()
            print("  predicting values")
            predictions = self.predict()
            print(
                f"  train evaluation: {self._evaluate(self.preferences, predictions)}"
            )
            if test_preferences is not None:
                print(
                    f"  test evaluation: {self._evaluate(test_preferences, predictions)}"
                )
