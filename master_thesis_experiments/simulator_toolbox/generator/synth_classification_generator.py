import random
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from master_thesis_experiments.simulator_toolbox.enums import GeneratorType
from master_thesis_experiments.simulator_toolbox.generator.base import Generator
from master_thesis_experiments.simulator_toolbox.utils import (
    compute_default_columns_names,
    generate_triangular_matrix,
    get_logger,
    is_pd,
    nearest_pd,
)

logger = get_logger(__name__)


class SynthClassificationGenerator(Generator):
    """
    Generator implementation to split data from an external dataset
    """

    def __init__(self, n_features, n_outputs, n_classes):
        super().__init__(
            generator_type=GeneratorType.SYNTH,
            name='interchanging_rbf_generator',
        )

        self.size = n_features
        self.n_classes = n_classes
        self.columns_names = compute_default_columns_names(
            n_features, n_outputs
        )

        self.mean_values = []
        self._cov_values = []
        self.covariance_matrices = np.zeros((self.n_classes, self.size, self.size))
        self.classes = list(range(self.n_classes))

        self.current_interchange_prob = 0.00
        self.current_interchanging_rbfs = []
        self.switches = 0
        self.switched = False

    @property
    def cov_values(self):
        """
        cov values property
        """
        return self._cov_values

    @cov_values.setter
    def cov_values(self, value):
        """
        cov values setter
        """
        self._cov_values = value

        for i, init_values in enumerate(self._cov_values):
            triangular_matrix = generate_triangular_matrix(
                init_values, self.size
            )

            self.covariance_matrices[i] = nearest_pd(triangular_matrix)

    def generate(self, n_samples):
        """
        Generate data in batches
        """
        if len(self.cov_values) == 0:
            raise ValueError("insert cov values")

        self.is_generating = True

        data = []

        random_vars = []

        for mean, covariance_matrix in zip(
                self.mean_values, self.covariance_matrices
        ):

            if not is_pd(covariance_matrix):
                covariance_matrix = nearest_pd(covariance_matrix)

            random_var = multivariate_normal(mean=mean, cov=covariance_matrix)

            random_vars.append(random_var)

        for _ in range(n_samples):

            random_vars = []

            for mean, covariance_matrix in zip(
                    self.mean_values, self.covariance_matrices
            ):
                random_var = multivariate_normal(
                    mean=mean, cov=covariance_matrix
                )

                random_vars.append(random_var)

            selected = random.randint(0, len(random_vars) - 1)

            entry = random_vars[selected].rvs()

            entry = np.append(entry, selected)

            data.append(entry)

        self.is_generating = False

        batch_df = pd.DataFrame(data, columns=self.columns_names)

        return deepcopy(batch_df)
