import csv
import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.active_learning.random_sampling_v2 import (
    RandomSamplingStrategyV2,
)
from master_thesis_experiments.active_learning.weighted_sampling import (
    WeightedSamplingStrategy,
)
from master_thesis_experiments.adaptation.density_estimation import (
    MultivariateNormalEstimator,
)
from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import (
    SynthClassificationGenerator,
)
from master_thesis_experiments.simulator_toolbox.simulation.base import Simulation
from master_thesis_experiments.simulator_toolbox.utils import (
    get_logger,
    get_root_level_dir,
)

logger = get_logger(__name__)

PERTURBATION_TYPE = ["mean", "combination"]
PERTURBATION_INTENSITY = ["small", "large"]


def generate_mean_values(min_distance, n_features, n_classes):
    mean_values = []
    while len(mean_values) < n_classes:
        means = np.random.uniform(0, 15, n_features)
        too_close = False
        for existing_mean in mean_values:
            distance = euclidean_distances([means], [existing_mean])[0][0]
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            mean_values.append(means)
    return mean_values


def check_means_distances(mean_values, min_distance):
    for concept in mean_values.keys():
        for i, mean in enumerate(mean_values[concept].values()):
            for j, other_mean in enumerate(mean_values[concept].values()):
                if i != j:
                    distance = euclidean_distances([mean], [other_mean])[0][0]
                    if distance < min_distance:
                        return False
    return True


class SynthClassificationSimulationV2(Simulation):
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

        self.clairvoyant_final_accuracy = None
        self.current_concept_extended = None
        self.test_set_size = test_set_size
        self.mean_values = []
        self.cov_values = []

        self.AL_accuracy = {}
        self.clairvoyant_accuracy = {}
        self.pre_AL_accuracy = None
        self.weights = None
        self.gamma = None
        self.alpha = None
        self.gamma_handler = None

    def generate_dataset(self, n_concepts, concept_size, last_concept_size):
        """
        This method generates the dataset
        """
        logger.debug("Generating the dataset...")

        n = self.generator.size
        triangular_size = int(n * (n + 1) / 2)

        theta_values = [np.pi, np.pi / 2, np.pi / 4, 3 * np.pi / 4]

        self.mean_values = generate_mean_values(
            min_distance=4,
            n_features=self.generator.size,
            n_classes=self.generator.n_classes,
        )

        self.cov_values = [
            np.random.uniform(4, 7, triangular_size)
            for _ in range(self.generator.n_classes)
        ]

        self.generator.mean_values = self.mean_values
        self.generator.cov_values = self.cov_values

        alpha = 2
        # the higher the alpha, the more balanced the prior probabilities
        self.generator.prior_probs = np.random.dirichlet(
            alpha * np.ones(self.generator.n_classes)
        )

        # generate a set of perturbations for each class, where the set size is
        # equal to the number of concepts. Some classes will have a small perturbation, while
        # others will have a large perturbation.
        # A perturbation can consist of adding noise to the
        # means or to the cov matrices of the classes or to rotate the cov matrices, or
        # a combination of these.

        perturbations = []

        valid_means = False

        mean_values = None
        covariance_matrices = None

        while valid_means is not True:

            mean_values = {}
            covariance_matrices = {}

            for class_index in range(self.generator.n_classes):
                perturbations.append([])

                class_perturbation_type = np.random.choice(PERTURBATION_TYPE, size=1)[0]
                class_perturbation_intensity = np.random.choice(
                    PERTURBATION_INTENSITY, size=1
                )[0]

                for _ in range(4):
                    perturbation = {}

                    if (
                            class_perturbation_type == "mean"
                            and class_perturbation_intensity == "small"
                    ):
                        perturbation["type"] = "mean"
                        perturbation["mean"] = np.random.normal(
                            scale=1, size=self.generator.size
                        )

                        cov_noise = np.random.uniform(-1, 1, triangular_size)

                        perturbation['cov'] = cov_noise

                    elif (
                            class_perturbation_type == "mean"
                            and class_perturbation_intensity == "large"
                    ):
                        perturbation["type"] = "mean"
                        perturbation["mean"] = np.random.normal(
                            scale=5, size=self.generator.size
                        )

                        cov_noise = np.random.uniform(-1, 1, triangular_size)
                        perturbation["cov"] = cov_noise

                    elif (
                            class_perturbation_type == "combination"
                            and class_perturbation_intensity == "small"
                    ):
                        perturbation["type"] = "combination"
                        perturbation["mean"] = np.random.normal(
                            scale=1, size=self.generator.size
                        )

                        cov_noise = np.random.uniform(-2, 2, triangular_size)

                        perturbation["cov"] = cov_noise
                        perturbation["theta"] = np.random.choice(theta_values, size=1)

                    else:
                        perturbation["type"] = "combination"
                        perturbation["mean"] = np.random.normal(
                            scale=5, size=self.generator.size
                        )

                        cov_noise = np.random.uniform(-5, 5, triangular_size)

                        perturbation["cov"] = cov_noise
                        perturbation["theta"] = np.random.choice(theta_values, size=1)

                    perturbations[class_index].append(perturbation)

            for i in range(n_concepts):

                mean_values[i] = {}
                covariance_matrices[i] = {}

                self.concept_mapping["concept_" + str(i)] = {}

                for class_ in range(self.generator.n_classes):

                    concept_perturbation = np.random.choice(perturbations[class_])

                    if concept_perturbation["type"] == "mean":

                        self.generator.mean_values[class_] = (
                                self.mean_values[class_] + concept_perturbation["mean"]
                        )

                        self.generator.cov_values[class_] = (
                                self.cov_values[class_] + concept_perturbation["cov"]
                        )

                        mean_values[i][class_] = (
                                self.mean_values[class_] + concept_perturbation["mean"]
                        )
                        covariance_matrices[i][class_] = self.generator.covariance_matrices[class_]

                    else:

                        self.generator.mean_values[class_] = (
                                self.mean_values[class_] + concept_perturbation["mean"]
                        )
                        self.generator.cov_values[class_] = (
                                self.cov_values[class_] + concept_perturbation["cov"]
                        )

                        dims = np.random.choice(
                            list(range(self.generator.size)), size=2, replace=False
                        )
                        self.generator.rotate(
                            dims[0], dims[1], concept_perturbation["theta"], class_
                        )

                        mean_values[i][class_] = (
                                self.mean_values[class_] + concept_perturbation["mean"]
                        )
                        covariance_matrices[i][class_] = self.generator.covariance_matrices[class_]

            valid_means = check_means_distances(mean_values, min_distance=4)

        for i in range(n_concepts):

            for class_ in range(self.generator.n_classes):
                self.generator.mean_values[class_] = mean_values[i][class_]
                self.generator.covariance_matrices[class_] = covariance_matrices[i][class_]

                self.concept_mapping["concept_" + str(i)][
                    "class_" + str(class_)
                    ] = multivariate_normal(
                    self.generator.mean_values[class_],
                    self.generator.covariance_matrices[class_],
                )

                # modify means and covs types for storing
                mean_values[i][class_] = mean_values[i][class_].tolist()
                covariance_matrices[i][class_] = covariance_matrices[i][class_].tolist()

            self.prior_probs_per_concept.append(self.generator.prior_probs.tolist())

            if i != n_concepts - 1:
                dataset = self.generator.generate(concept_size)
                generated_classes = dataset["y_0"].unique()

                while len(generated_classes) != self.generator.n_classes:
                    logger.debug(
                        "Regenerating dataset, some class did not generate any sample"
                    )
                    dataset = self.generator.generate(concept_size)
                    generated_classes = dataset["y_0"].unique()
            else:
                dataset = self.generator.generate(last_concept_size)
                generated_classes = dataset["y_0"].unique()

                while len(generated_classes) != self.generator.n_classes:
                    logger.debug(
                        "Regenerating dataset, some class did not generate any sample"
                    )
                    dataset = self.generator.generate(last_concept_size)
                    generated_classes = dataset["y_0"].unique()

                if self.test_set is None:
                    self.test_set = self.generator.generate(self.test_set_size)
                    self.test_set = DataProvider("test_set", self.test_set)

                    self.current_concept_extended = self.generator.generate(N_SAMPLES)
                    self.current_concept_extended = DataProvider(
                        "current_concept_extended", self.current_concept_extended
                    )

            # scaler.fit_transform(dataset)
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
            "means_per_concept": mean_values,
            "covs_per_concept": covariance_matrices,
            "n_classes": N_CLASSES,
            "test_set_size": self.test_set_size,
        }

    def run(self,gamma_handler, alpha, gamma):
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

        self.alpha = alpha
        self.gamma = gamma
        self.gamma_handler= gamma_handler

        scaler = preprocessing.StandardScaler()

        classifier = LogisticRegression(
            multi_class="multinomial", solver="sag", random_state=42
        )

        current_concept = deepcopy(self.concepts[-1].generated_dataset)

        X, y = self.concepts[-1].get_split_dataset()

        X = scaler.fit_transform(X)

        classifier.fit(X=X, y=y)

        test_set = deepcopy(self.test_set)
        data = test_set.generated_dataset.values
        X = data[:, :-1]
        test_set.generated_dataset[test_set.generated_dataset.columns[:-1]] = scaler.fit_transform(X)
        X_test, y_test = test_set.get_split_dataset()
        self.pre_AL_accuracy = classifier.score(X_test, y_test)

        n_samples = self.n_samples

        # Clairvoyant
        while n_samples > 0:
            n_selected_samples = self.n_samples - n_samples + 1

            current_concept = pd.concat(
                (
                    current_concept,
                    self.current_concept_extended.generated_dataset.iloc[
                        [n_samples - 1]
                    ],
                ),
                ignore_index=True,
            )

            scaled_current_concept = deepcopy(current_concept)

            data = scaled_current_concept.values
            X = data[:, :-1]
            scaled_current_concept[scaled_current_concept.columns[:-1]] = scaler.fit_transform(X)

            X_clrv, y_clrv = (
                scaled_current_concept[scaled_current_concept.columns[:-1]],
                scaled_current_concept[scaled_current_concept.columns[-1]],
            )

            classifier.fit(X=X_clrv.to_numpy(), y=y_clrv.to_numpy())

            self.clairvoyant_accuracy[
                n_selected_samples
            ] = classifier.score(X_test, y_test)

            n_samples -= 1

        current_concept = pd.concat(
            (deepcopy(self.concepts[-1].generated_dataset), self.current_concept_extended.generated_dataset),
            ignore_index=True)

        data = current_concept.values
        X = data[:, :-1]
        current_concept[current_concept.columns[:-1]] = scaler.fit_transform(X)

        X_clrv, y_clrv = (
            current_concept[current_concept.columns[:-1]],
            current_concept[current_concept.columns[-1]],
        )

        classifier.fit(X=X_clrv.to_numpy(), y=y_clrv.to_numpy())

        self.clairvoyant_final_accuracy = classifier.score(X_test, y_test)

        # AL strategy
        for strategy in self.strategies:
            if strategy.__name__ == 'WeightedSamplingStrategy':
                strategy_instance: WeightedSamplingStrategy = strategy(
                    concept_mapping=deepcopy(self.concept_mapping),
                    concept_list=deepcopy(self.concepts),
                    n_samples=self.n_samples,
                    estimator_type=self.estimator_type,
                    prior_probs=deepcopy(self.prior_probs_per_concept[0]),
                    gamma_handler=gamma_handler,
                    alpha=alpha,
                    gamma=gamma
                )

            else:
                strategy_instance: BaseStrategy = strategy(
                    concept_mapping=deepcopy(self.concept_mapping),
                    concept_list=deepcopy(self.concepts),
                    n_samples=self.n_samples,
                    estimator_type=self.estimator_type,
                    prior_probs=deepcopy(self.prior_probs_per_concept[0]),
                )
            self.strategy_instances.append(strategy_instance)

            strategy_instance.initialize()

            n_samples = self.n_samples

            while n_samples > 0:
                X_new, y_new = strategy_instance.run()
                classifier.fit(X=X_new, y=y_new)

                n_selected_samples = self.n_samples - n_samples + 1
                self.AL_accuracy[
                    (strategy_instance.name, n_selected_samples)
                ] = classifier.score(X_test, y_test)

                n_samples -= 1

            self.selected_samples_per_strategy[
                strategy_instance.name
            ] = strategy_instance.all_selected_samples

            if isinstance(strategy_instance, WeightedSamplingStrategy):
                self.weights = strategy_instance.weights
                self.weights = pd.concat(
                    (
                        self.weights,
                        strategy_instance.weighting_handler.selected_samples_weights,
                    ),
                    ignore_index=False,
                )

    def store_results(self, experiment_index):
        simulation_results_dir = self.simulation_results_dir + "_gamma_handler:" + str(self.gamma_handler) + "-" + "alpha:" + str(self.alpha) + "-" + "gamma:" + str(
            self.gamma)
        concepts_path = Path(simulation_results_dir + "/" + str(experiment_index))
        concepts_path.mkdir(parents=True, exist_ok=True)

        # Save concepts
        for concept in self.concepts:
            concept_path = concepts_path / str(concept.name + ".csv")
            concept.generated_dataset.to_csv(concept_path, index=False)

        # Save test set
        test_set_path = Path(
            simulation_results_dir + "/" + str(experiment_index) + "/test_set.csv"
        )
        self.test_set.generated_dataset.to_csv(test_set_path, index=False)

        # Save pre-AL accuracy
        pre_AL_accuracy_path = Path(
            simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/pre_AL_accuracy.csv"
        )

        with open(pre_AL_accuracy_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([self.pre_AL_accuracy])

        # Save Clairvoyant final accuracy
        clairvoyant_final_accuracy_path = Path(
            simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/clairvoyant_final_accuracy.csv"
        )

        with open(clairvoyant_final_accuracy_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([self.clairvoyant_final_accuracy])

        # Save AL accuracy
        for key, item in self.AL_accuracy.items():
            AL_accuracy_path = Path(
                simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + str(key[0])
                + "/"
                + str(key[1])
                + "_samples.csv"
            )
            AL_accuracy_path.parent.mkdir(parents=True, exist_ok=True)

            with open(AL_accuracy_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([item])

        columns = self.concepts[0].generated_dataset.columns
        for strategy_name, samples in self.selected_samples_per_strategy.items():
            selected_samples_path = Path(
                simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + str(strategy_name)
                + "/"
                + "selected_samples.csv"
            )
            pd.DataFrame(samples, columns=columns).to_csv(
                selected_samples_path, index=False
            )

        # Save clairvoyant accuracy
        for key, item in self.clairvoyant_accuracy.items():
            clairvoyant_accuracy_path = Path(
                simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + "clairvoyant"
                + "/"
                + str(key)
                + "_samples.csv"
            )
            clairvoyant_accuracy_path.parent.mkdir(parents=True, exist_ok=True)

            with open(clairvoyant_accuracy_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([item])

        # save weights
        if self.weights is not None:
            weights_path = Path(
                simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/weights.csv"
            )
            self.weights.to_csv(weights_path, index=False)

        # save generation metadata
        metadata_file = (
                simulation_results_dir + "/" + str(experiment_index) + "/metadata.json"
        )
        with open(metadata_file, "w") as metadata_file:
            json.dump(self.metadata, metadata_file)

    def soft_reset(self):
        self.generator.reset()

        self.metadata = None
        self.concept_mapping = {}
        self.strategy_instances = []
        self.concepts = []
        self.prior_probs_per_concept = []
        self.selected_samples_per_strategy = {}

        self.test_set = None

        self.AL_accuracy = {}
        self.pre_AL_accuracy = 0.0
        self.weights = None


if __name__ == "__main__":
    N_EXPERIMENTS = 20

    N_SAMPLES = 200

    N_FEATURES = 2
    N_CLASSES = 10

    N_CONCEPTS = 5
    CONCEPT_SIZE = 1000
    LAST_CONCEPT_SIZE = 13

    TEST_SET_SIZE = 300

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5]
    gamma_handler = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    simulation = SynthClassificationSimulationV2(
        name="synth_classification_fixed_dataset_and_samples_v2",
        generator=SynthClassificationGenerator(
            n_features=N_FEATURES, n_outputs=1, n_classes=N_CLASSES
        ),
        strategies=[
            WeightedSamplingStrategy,
            RandomSamplingStrategyV2,
        ],
        results_dir=get_root_level_dir("results"),
        n_samples=N_SAMPLES,
        estimator_type=MultivariateNormalEstimator,
        test_set_size=TEST_SET_SIZE,
    )

    for g_h, a, g in itertools.product(gamma_handler, alphas, gammas):

        for experiment in tqdm(range(N_EXPERIMENTS)):
            simulation.generate_dataset(
                n_concepts=N_CONCEPTS,
                concept_size=CONCEPT_SIZE,
                last_concept_size=LAST_CONCEPT_SIZE,
            )

            # simulation.store_concepts(experiment)

            simulation.run(g_h, a, g)

            simulation.store_results(experiment)

            simulation.soft_reset()
