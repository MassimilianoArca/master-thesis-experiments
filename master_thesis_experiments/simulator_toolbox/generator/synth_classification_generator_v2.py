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


class SemicircleRandomVariable:
    def __init__(self, radius, center, noise):
        self.radius = radius
        self.center = center
        self.noise = noise

    def rvs(self, theta, factor):
        x_1 = (
            self.radius * np.cos(theta)
            + self.center[0]
            + np.random.uniform(-self.noise, self.noise)
        )
        x_2 = (
            self.radius * np.sin(theta) * factor
            + self.center[1]
            + np.random.uniform(-self.noise, self.noise)
        )

        return x_1, x_2


class SynthClassificationGeneratorV2:
    """
    Generator implementation to split data from an external dataset
    """

    def __init__(self, n_features, n_outputs, n_classes):
        self.name = "semicircle_generator"

        self.size = n_features
        self.n_classes = n_classes
        self.columns_names = compute_default_columns_names(n_features, n_outputs)

        self.radius_values = []
        self.centers = []
        self.factors = []
        self.noises = []
        self.classes = list(range(self.n_classes))
        self.prior_probs = []

    def generate(self, n_samples):
        """
        Generate data in batches
        """

        data = []
        random_vars = []

        for radius, center, noise in zip(self.radius_values, self.centers, self.noises):
            random_var = SemicircleRandomVariable(radius, center, noise)

            random_vars.append(random_var)

        for _ in range(n_samples):
            selected = np.random.choice(
                range(len(self.prior_probs)), p=self.prior_probs
            )

            # generate row for the class label 'selected', which is a
            # float randomly selected in [0, len(random_vars) - 1]
            theta = np.random.uniform(0, np.pi, size=1)
            entry = random_vars[selected].rvs(
                theta=theta, factor=self.factors[selected]
            )

            entry = np.append(entry, selected)

            data.append(entry)

        batch_df = pd.DataFrame(data, columns=self.columns_names)

        return deepcopy(batch_df)

    def reset(self):
        self.prior_probs = []
        self.factors = []
        self.noises = []
        self.centers = []
        self.radius_values = []
