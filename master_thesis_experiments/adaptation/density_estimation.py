from abc import abstractmethod
from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.covariance import OAS
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


class DensityEstimator(object):
    """
    Density estimator abstract class.
    """

    def __init__(self, concept_id: str = ''):
        super().__init__()
        self.dataset = None
        self.concept_id = concept_id

    def fit(self, dataset: np.ndarray):
        """
        Fit the Density estimator
        """
        self.dataset = dataset

    @abstractmethod
    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Return the likelihoods of the given samples.
        """


class ClairvoyantNormalEstimator(DensityEstimator):
    """
    A normal density estimator in which mean and covariance are already
    known. It defaults to zero mean and eye covariance matrix if those
    values are not provided.
    """

    def __init__(
            self,
            mean: Optional[np.ndarray] = None,
            cov: Optional[np.ndarray] = None,
            concept_id: str = '',
    ):
        super().__init__(concept_id)
        self.mean = mean
        self.cov = cov

    def fit(self, dataset):
        dim = None
        super().fit(dataset)

        if len(dataset.shape) > 1:
            dim = dataset.shape[1]
        if self.mean is None:
            self.mean = np.zeros(dim)
        if self.cov is None:
            self.cov = np.eye(dim)

    def pdf(self, X: np.ndarray) -> np.ndarray:
        return multivariate_normal.pdf(X, mean=self.mean, cov=self.cov, allow_singular=True)


class MultivariateNormalEstimator(ClairvoyantNormalEstimator):
    """
    A Multivariate Normal density estimator employing OAS for covariance
    estimation.
    """

    def __init__(self, concept_id: str = ''):
        super().__init__(concept_id=concept_id)
        self.oas = OAS()
        self.mean = None
        self.cov = None

    def fit(self, dataset: np.ndarray):
        # super().fit(dataset)
        self.oas.fit(dataset)

        if self.mean is None:
            self.mean = self.oas.location_

        if self.cov is None:
            self.cov = self.oas.covariance_


class KernelDensityEstimator(DensityEstimator):
    """
    A kernel density estimator wrapping the scikit-learn implementation.
    """

    def __init__(self, concept_id: str = ''):
        super().__init__(concept_id)
        self.kernel_density = KernelDensity()

    # Needs a refit once loaded from a snapshot since
    # the tree_ attribute of KernelDensity() cannot be
    # snapshotted
    def fit(self, dataset):
        super().fit(dataset)
        self.kernel_density.fit(dataset)

    def pdf(self, X: np.ndarray):
        return np.exp(self.kernel_density.score_samples(X))


class GaussianMixtureEstimator(DensityEstimator):
    """
    A Mixture density estimator wrapping the
    scikit-learn implementation.
    """

    def __init__(self, concept_id: str = ''):
        super().__init__(concept_id)
        self.density = GaussianMixture(n_components=5, covariance_type='full')

    def fit(self, dataset):
        super().fit(dataset)
        self.density.fit(dataset)

    def pdf(self, X: np.ndarray):
        return np.exp(self.density.score_samples(X))


class MultipleEstimator(DensityEstimator):
    """
    Combining n multivariate normal estimators
    where n is the number of output classes
    A new estimator for each
    output class is used
    """

    def __init__(self, concept_id: str = ''):
        super().__init__(concept_id)

        self.classes = np.array([])
        self.estimators = {}

    def fit(self, dataset: np.ndarray):
        super().fit(dataset)

        self.classes = np.unique(dataset[:, -1])

        # for each output class
        # fit an estimator
        for output_class in self.classes:
            # filter data on output class
            filtered_data = [
                data for data in dataset if data[-1] == output_class
            ]

            # if there are some samples
            if len(filtered_data) >= 2:  # hard-coded for now
                estimator = MultivariateNormalEstimator()
                estimator.fit(np.array(filtered_data)[:, :-1])
                self.estimators[output_class] = estimator

            # temporary return a fixed Normal Distribution
            # to prevent error in the estimator
            else:
                estimator = ClairvoyantNormalEstimator()
                estimator.fit(np.array(filtered_data))
                self.estimators[output_class] = estimator

    def pdf(self, X: np.ndarray):
        result = np.array([])
        for sample in X:
            # select the correct estimator and divide
            # by number of classes
            # NOTE: assuming balanced classes for now
            result = np.append(
                result,
                self.estimators[sample[-1]].pdf(np.array([sample]))
                / len(self.classes),
            )
        return result
