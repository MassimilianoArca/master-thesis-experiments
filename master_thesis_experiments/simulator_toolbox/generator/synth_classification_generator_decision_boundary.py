import random
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

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


class SynthClassificationGeneratorDecisionBoundary:
    """
    Generator implementation to split data from an external dataset
    """

    def __init__(self, n_features, n_outputs, n_classes):
        self.name = "decision_boundary_generator"

        self.size = n_features
        self.n_classes = n_classes
        self.columns_names = compute_default_columns_names(n_features, n_outputs)

    def generate(self, n_samples):
        """
        Generate data in batches
        """

        X, _ = make_blobs(
            n_samples=n_samples,
            centers=self.n_classes,
            n_features=self.size,
            cluster_std=3,
            random_state=42
        )

        batch_df = pd.DataFrame(X, columns=self.columns_names)

        return deepcopy(batch_df)
