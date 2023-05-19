import random
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.active_learning.label_spreading import LabelSpreadingStrategy
from master_thesis_experiments.active_learning.random_sampling import (
    RandomSamplingStrategy,
)
from master_thesis_experiments.active_learning.uncertainty_spreading import (
    UncertaintySpreadingStrategy,
)
from master_thesis_experiments.adaptation.density_estimation import (
    MultivariateNormalEstimator,
)
from master_thesis_experiments.handlers.importance_weights import IWHandler
from master_thesis_experiments.handlers.joint_probability import JointProbabilityHandler
from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import (
    SynthClassificationGenerator,
)
from master_thesis_experiments.simulator_toolbox.model.base import Model
from master_thesis_experiments.simulator_toolbox.simulation.base import Simulation
from master_thesis_experiments.simulator_toolbox.utils import (
    get_logger,
    get_root_level_dir,
)

logger = get_logger(__name__)


def generate_mean_values(min_distance, n_features, n_classes):
    mean_values = []
    while len(mean_values) < n_classes:
        means = np.random.uniform(0, 10, n_features)
        too_close = False
        for existing_mean in mean_values:
            distance = euclidean_distances([means], [existing_mean])[0][0]
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            mean_values.append(means)
    return mean_values


class SynthClassificationSimulation(Simulation):
    def __init__(
        self,
        name,
        generator,
        strategies,
        results_dir,
        n_samples,
        estimator_type,
        test_set_size,
    ):
        super().__init__(
            name, generator, strategies, results_dir, n_samples, estimator_type
        )

        self.test_set_size = test_set_size
        self.mean_values = []
        self.cov_values = []

    def generate_dataset(self, n_concepts, concept_size, last_concept_size):
        """
        This method generates the dataset
        """
        logger.debug("Generating the dataset...")

        scaler = preprocessing.StandardScaler()

        n = self.generator.size
        triangular_size = int(n * (n + 1) / 2)

        theta_values = [np.pi, np.pi / 2, np.pi / 4, 3 * np.pi / 4]

        self.mean_values = generate_mean_values(
            min_distance=4.5,
            n_features=self.generator.size,
            n_classes=self.generator.n_classes,
        )

        self.cov_values = [
            np.random.uniform(2, 5, triangular_size)
            for _ in range(self.generator.n_classes)
        ]

        self.generator.mean_values = self.mean_values
        self.generator.cov_values = self.cov_values

        alpha = 2
        # the higher the alpha, the more balanced the prior probabilities
        self.generator.prior_probs = np.random.dirichlet(
            alpha * np.ones(self.generator.n_classes)
        )

        for i in range(n_concepts):
            self.concept_mapping["concept_" + str(i)] = {}

            # perturbation of the means and cov matrices
            mean_noises = [
                np.random.normal(scale=1, size=self.generator.size)
                for _ in range(self.generator.n_classes)
            ]

            # choose two random dimensions
            dims = np.random.choice(
                list(range(self.generator.size)), size=2, replace=False
            )

            # rotate a random class cov matrix per concept
            class_matrix = np.random.choice(
                list(range(len(self.generator.covariance_matrices))),
                size=1,
                replace=False,
            )
            theta = np.random.choice(theta_values, size=1)

            self.generator.rotate(dims[0], dims[1], theta, class_matrix)

            for class_ in range(self.generator.n_classes):
                self.generator.mean_values[class_] = (
                    self.mean_values[class_] + mean_noises[class_]
                )
                self.generator.cov_values[class_] = self.cov_values[class_]

                self.concept_mapping["concept_" + str(i)][
                    "class_" + str(class_)
                ] = multivariate_normal(
                    self.generator.mean_values[class_],
                    self.generator.covariance_matrices[class_],
                )

            # perturbation of the prior probabilities
            # self.generator.prior_probs = something
            self.prior_probs_per_concept.append(self.generator.prior_probs.tolist())

            if i != n_concepts - 1:
                dataset = self.generator.generate(concept_size)
                # test set will be of the first concept distribution
                if self.test_set is None:
                    self.test_set = self.generator.generate(self.test_set_size)
            else:
                dataset = self.generator.generate(last_concept_size)

            scaler.fit_transform(dataset)
            self.concepts.append(DataProvider("concept_" + str(i), dataset))

        self.metadata = {
            "dataset_name": self.generator.name,
            "past_dataset_size": (n_concepts - 1) * concept_size,
            "task": "classification",
            "type": "synth",
            "n_concepts": n_concepts,
            "concept_size": concept_size,
            "last_concept_size": last_concept_size,
            "prior_probs_per_concept": self.prior_probs_per_concept,
            "n_samples": self.n_samples,
            "means": [means.tolist() for means in self.generator.mean_values],
            "covs": [covs.tolist() for covs in self.generator.covariance_matrices],
        }

    def run(self):
        """
        iw_handler = IWHandler(
            concept_mapping=self.concept_mapping,
            concept_list=self.concepts,
            estimator_type=self.estimator_type,
            prior_class_probabilities=self.prior_probs_per_concept,
        )

        # compute true weights
        self.true_weights = iw_handler.run_true_weights().tolist()

        # compute pre-AL weights
        self.pre_AL_weights = iw_handler.run_weights().tolist()

        iw_handler.soft_reset()
        """

        classifier = Model(name="Logistic Regression", ml_model=LogisticRegression())

        joint_prob_handler = JointProbabilityHandler(
            concept_mapping=self.concept_mapping,
            concept_list=self.concepts,
            estimator_type=self.estimator_type,
            classifier=classifier,
            test_set=self.test_set,
            prior_class_probabilities=self.prior_probs_per_concept[-1],
        )

        joint_prob_handler.initialize()

        self.true_p_y_given_x = joint_prob_handler.compute_true_conditional()
        self.true_p_x = joint_prob_handler.true_p_x

        self.pre_AL_p_y_given_x = joint_prob_handler.estimate_conditional()
        self.p_x = joint_prob_handler.estimate_input()

        for strategy in self.strategies:
            strategy_instance: BaseStrategy = strategy(
                concept_mapping=deepcopy(self.concept_mapping),
                concept_list=deepcopy(self.concepts),
                n_samples=self.n_samples,
                estimator_type=self.estimator_type,
            )
            self.strategy_instances.append(strategy_instance)

            strategy_instance.initialize()
            strategy_instance.estimate_new_concept()

            n_samples = self.n_samples
            while n_samples > 0:
                new_concept_list = strategy_instance.run()
                """
                # compute post-AL weights
                iw_handler.concept_list = new_concept_list

                n_selected_samples = self.n_samples - n_samples + 1
                self.strategy_post_AL_weights[
                    (strategy_instance.name, n_selected_samples)
                ] = iw_handler.run_weights().tolist()

                iw_handler.soft_reset()
                """

                joint_prob_handler.concept_list = new_concept_list

                n_selected_samples = self.n_samples - n_samples + 1
                self.p_y_given_x[
                    (strategy_instance.name, n_selected_samples)
                ] = joint_prob_handler.estimate_conditional()

                n_samples -= 1

            self.selected_samples_per_strategy[
                strategy_instance.name
            ] = strategy_instance.all_selected_samples


if __name__ == "__main__":
    N_EXPERIMENTS = 3

    N_SAMPLES = 100

    N_FEATURES = 4
    N_CLASSES = 5

    N_CONCEPTS = 5
    CONCEPT_SIZE = 1000
    LAST_CONCEPT_SIZE = 30
    TEST_SET_SIZE = 300

    simulation = SynthClassificationSimulation(
        name="synth_classification_fixed_dataset_and_samples",
        generator=SynthClassificationGenerator(
            n_features=N_FEATURES, n_outputs=1, n_classes=N_CLASSES
        ),
        strategies=[
            UncertaintySpreadingStrategy,
            LabelSpreadingStrategy,
            RandomSamplingStrategy,
        ],
        results_dir=get_root_level_dir("results"),
        n_samples=N_SAMPLES,
        estimator_type=MultivariateNormalEstimator,
        test_set_size=TEST_SET_SIZE,
    )

    for experiment in tqdm(range(N_EXPERIMENTS)):
        simulation.generate_dataset(
            n_concepts=N_CONCEPTS,
            concept_size=CONCEPT_SIZE,
            last_concept_size=LAST_CONCEPT_SIZE,
        )

        # simulation.store_concepts(experiment)

        simulation.run()

        simulation.store_results(experiment)

        simulation.soft_reset()
