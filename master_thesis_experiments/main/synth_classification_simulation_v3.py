import csv
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.metrics import euclidean_distances
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

from master_thesis_experiments.active_learning.base import BaseStrategyV3
from master_thesis_experiments.active_learning.entropy_diversity_sampling import (
    EntropyDiversitySamplingStrategy,
)
from master_thesis_experiments.active_learning.entropy_sampling import (
    EntropySamplingStrategy,
)
from master_thesis_experiments.active_learning.random_sampling_v3 import (
    RandomSamplingStrategyV3,
)
from master_thesis_experiments.adaptation.density_estimation import MultivariateNormalEstimator
from master_thesis_experiments.handlers.importance_weights import IWHandler

from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import (
    SynthClassificationGenerator,
)
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator_decision_boundary import \
    SynthClassificationGeneratorDecisionBoundary
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator_v2 import (
    SynthClassificationGeneratorV2,
)
from master_thesis_experiments.simulator_toolbox.utils import (
    get_root_level_dir,
    get_logger,
)

logger = get_logger(__name__)

PERTURBATION_TYPE_V3 = ["radius", "center", "noise", "combination"]
PERTURBATION_INTENSITY_V3 = ["small", "large"]

PERTURBATION_TYPE_V2 = ["mean", "covariance", "combination"]
PERTURBATION_INTENSITY_V2 = ["small", "large"]


def generate_mean_values(min_distance, n_features, n_classes):
    mean_values = []
    while len(mean_values) < n_classes:
        means = np.random.uniform(-5, 5, n_features)
        too_close = False
        for existing_mean in mean_values:
            distance = euclidean_distances([means], [existing_mean])[0][0]
            if distance < min_distance:
                too_close = True
                break
        if not too_close:
            mean_values.append(means)
    return mean_values


def create_perturbations_v3(n_classes, n_perturbations):
    perturbations = []

    for class_index in range(n_classes):
        perturbations.append([])

        class_perturbation_type = np.random.choice(PERTURBATION_TYPE_V3, size=1)[0]
        class_perturbation_intensity = np.random.choice(
            PERTURBATION_INTENSITY_V3, size=1
        )[0]

        for _ in range(n_perturbations):
            perturbation = {
                "type": class_perturbation_type,
                "intensity": class_perturbation_intensity,
            }

            if (
                    class_perturbation_type == "radius"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "radius": np.random.uniform(0.1, 0.3, size=1),
                    "center": 0,
                    "noise": 0,
                }

            elif (
                    class_perturbation_type == "center"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "radius": 0,
                    "center": np.random.uniform(-0.7, 0.7, size=2),
                    "noise": 0,
                }

            elif (
                    class_perturbation_type == "noise"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "radius": 0,
                    "center": 0,
                    "noise": np.random.uniform(0.01, 0.05, size=1),
                }

            elif (
                    class_perturbation_type == "combination"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "radius": np.random.uniform(0.1, 0.3, size=1),
                    "center": np.random.uniform(-0.2, 0.2, size=2),
                    "noise": np.random.uniform(0.01, 0.05, size=1),
                }

            elif (
                    class_perturbation_type == "radius"
                    and class_perturbation_intensity == "large"
            ):
                perturbation["value"] = {
                    "radius": np.random.uniform(0.3, 0.6, size=1),
                    "center": 0,
                    "noise": 0,
                }

            elif (
                    class_perturbation_type == "center"
                    and class_perturbation_intensity == "large"
            ):
                perturbation["value"] = {
                    "radius": 0,
                    "center": np.random.uniform(-3, 3, size=2),
                    "noise": 0,
                }

            elif (
                    class_perturbation_type == "noise"
                    and class_perturbation_intensity == "large"
            ):
                perturbation["value"] = {
                    "radius": 0,
                    "center": 0,
                    "noise": np.random.uniform(0.1, 0.2, size=1),
                }

            elif (
                    class_perturbation_type == "combination"
                    and class_perturbation_intensity == "large"
            ):
                perturbation["value"] = {
                    "radius": np.random.uniform(0.3, 0.6, size=1),
                    "center": np.random.uniform(-0.4, 0.4, size=2),
                    "noise": np.random.uniform(0.05, 0.1, size=1),
                }

            else:
                raise ValueError("Invalid perturbation type or intensity")

            perturbations[class_index].append(perturbation)

    return perturbations


def create_perturbations_v2(
        n_features, triangular_size, theta_values, n_classes, n_perturbations
):
    perturbations = []

    for class_index in range(n_classes):
        perturbations.append([])

        class_perturbation_type = np.random.choice(PERTURBATION_TYPE_V2, size=1)[0]
        class_perturbation_intensity = np.random.choice(
            PERTURBATION_INTENSITY_V2, size=1
        )[0]

        for _ in range(n_perturbations):
            perturbation = {"type": class_perturbation_type}

            if (
                    class_perturbation_type == "mean"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "mean": np.random.normal(scale=1, size=n_features),
                    "cov": 0,
                    "theta": 0,
                }

            elif (
                    class_perturbation_type == "mean"
                    and class_perturbation_intensity == "large"
            ):
                perturbation["value"] = {
                    "mean": np.random.normal(scale=2, size=n_features),
                    "cov": 0,
                    "theta": 0,
                }

            elif (
                    class_perturbation_type == "covariance"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "mean": 0,
                    "cov": np.random.uniform(-0.5, 0.5, triangular_size),
                    "theta": np.random.choice(theta_values, size=1),
                }

            elif (
                    class_perturbation_type == "covariance"
                    and class_perturbation_intensity == "large"
            ):
                perturbation["value"] = {
                    "mean": 0,
                    "cov": np.random.uniform(-2, 2, triangular_size),
                    "theta": np.random.choice(theta_values, size=1),
                }

            elif (
                    class_perturbation_type == "combination"
                    and class_perturbation_intensity == "small"
            ):
                perturbation["value"] = {
                    "mean": np.random.normal(scale=1, size=n_features),
                    "cov": np.random.uniform(-0.5, 0.5, triangular_size),
                    "theta": np.random.choice(theta_values, size=1),
                }

            else:
                perturbation["value"] = {
                    "mean": np.random.normal(scale=2, size=n_features),
                    "cov": np.random.uniform(-2, 2, triangular_size),
                    "theta": np.random.choice(theta_values, size=1),
                }

            perturbations[class_index].append(perturbation)

    return perturbations


def custom_boundary_1(x, y):
    return (x) ** 2 + y - 3


def custom_boundary_2(x, y):
    return -(x) ** 2 + y - x + 4


def rotate(x, y, theta):
    return [x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)]


def generate_data(data_size=10000,  # before imbalance, approx
                  x_min_max=2,
                  shape_param_1=1,  # try values like 3,5,7,[10] and see how graph changes
                  shape_param_2=0.5,  # try 5,10,15,20
                  shape_param_3=2,  # try between 1-5
                  shape_param_4=3,  # try 50,100,150,200
                  ):
    min_x, max_x = -x_min_max, x_min_max
    interval = (max_x - min_x) / data_size
    x_0 = np.arange(min_x, max_x, interval)
    y_line = np.clip((x_0 ** 3) / shape_param_4 - shape_param_1 * x_0 + shape_param_2 * np.sin(x_0 / shape_param_3),
                     -80, 80)

    x_1 = np.random.uniform(min(y_line) - 2, max(y_line) + 2, len(x_0))
    y = x_1 > y_line

    circle1 = np.where((x_0 - 0.5) ** 2 + (x_1 - 0.5) ** 2 < 1)
    y[circle1] = ~(np.mean(y[circle1]) > 0.5)

    circle2 = np.where((x_0 + 0.7) ** 2 + (x_1 + 1) ** 2 < 1)
    y[circle2] = ~(np.mean(y[circle2]) > 0.5)
    circle3 = np.where((x_0 - 1) ** 2 + (x_1 + 1) ** 2 < 1)
    y[circle3] = ~(np.mean(y[circle3]) > 0.5)

    circle4 = np.where((x_0 + 0.2) ** 2 + (x_1 - 0.2) ** 2 < 0.5)
    y[circle4] = ~(np.mean(y[circle4]) > 0.5)

    X = pd.DataFrame({"X_0": x_0, "X_1": x_1, "y_0": y})

    return X


class SynthClassificationSimulationV3:
    def __init__(
            self,
            name,
            strategies,
            result_dir,
            n_classes,
            n_queries,
            test_set_size,
    ):
        self.shape_param = None
        self.rotation_angle = None
        self.cov_values = None
        self.mean_values = None
        self.generator = None
        self.name = name
        self.strategies = strategies
        self.result_dir = result_dir
        self.n_classes = n_classes
        self.n_queries = n_queries
        self.test_set_size = test_set_size
        self.concept_list = []
        self.test_set = None
        self.AL_accuracy = {}
        self.clairvoyant_accuracy = {}
        self.pre_AL_accuracy = None
        self.current_concept_extended = None
        self.clairvoyant_final_accuracy = None
        self.metadata = None
        self.prior_probs = None
        self.strategy_instances = []
        self.selected_samples_per_strategy = {}
        self.prior_probs_per_concept = []
        self.concept_mapping = None
        self.ess_per_strategy = {}

        self.start_time = datetime.now()
        self.sim_id = self.start_time.strftime("%d-%m-%Y-%H:%M")

        path = result_dir + "/" + name + "/" + self.sim_id
        self.simulation_results_dir = path

    def generate_dataset(
            self,
            n_concepts,
            concept_size,
            last_concept_size,
            dataset_type,
    ):
        if dataset_type == "semicircles":
            self.generator = SynthClassificationGeneratorV2(
                n_features=2,
                n_outputs=1,
                n_classes=self.n_classes,
            )

            # base concept information
            # at each concept, these values will be perturbed
            centers = [np.random.uniform(-1, 1, size=2) for _ in range(self.n_classes)]
            radius_values = [
                np.random.uniform(0.1, 1.5, size=1) for _ in range(self.n_classes)
            ]
            factors = [random.choice([-1, 1]) for _ in range(self.n_classes)]
            noises = [
                np.random.uniform(0.1, 0.2, size=1) for _ in range(self.n_classes)
            ]

            for i in range(n_concepts):
                self.generator.reset()

                alpha = 1.5
                # the higher the alpha, the more balanced the prior probabilities
                self.prior_probs = np.random.dirichlet(alpha * np.ones(self.n_classes))

                # perturb concept
                perturbations = create_perturbations_v3(
                    self.n_classes, n_perturbations=4
                )

                for class_index in range(self.n_classes):
                    class_perturbation = np.random.choice(
                        perturbations[class_index], size=1
                    )[0]
                    perturbation_type = class_perturbation["type"]

                    if perturbation_type == "radius":
                        radius_values[class_index] += class_perturbation["value"][
                            "radius"
                        ]

                    elif perturbation_type == "center":
                        centers[class_index] += class_perturbation["value"]["center"]

                    elif perturbation_type == "noise":
                        noises[class_index] += class_perturbation["value"]["noise"]

                    elif perturbation_type == "combination":
                        radius_values[class_index] += class_perturbation["value"][
                            "radius"
                        ]
                        centers[class_index] += class_perturbation["value"]["center"]
                        noises[class_index] += class_perturbation["value"]["noise"]

                    else:
                        raise ValueError("Invalid perturbation type")

                self.generator.centers = centers
                self.generator.radius_values = radius_values
                self.generator.factors = factors
                self.generator.noises = noises
                self.generator.prior_probs = self.prior_probs

                if i != n_concepts - 1:
                    dataset = self.generator.generate(concept_size)
                    generated_classes = dataset["y_0"].unique()

                    while len(generated_classes) != self.n_classes:
                        logger.debug(
                            "Regenerating dataset, some class did not generate any sample"
                        )
                        dataset = self.generator.generate(concept_size)
                        generated_classes = dataset["y_0"].unique()

                # last concept
                else:
                    dataset = self.generator.generate(last_concept_size)
                    generated_classes = dataset["y_0"].unique()

                    while len(generated_classes) != self.n_classes:
                        logger.debug(
                            "Regenerating dataset, some class did not generate any sample"
                        )
                        dataset = self.generator.generate(last_concept_size)
                        generated_classes = dataset["y_0"].unique()

                    self.test_set = self.generator.generate(self.test_set_size)
                    self.current_concept_extended = self.generator.generate(
                        n_samples=10000
                    )
                    self.current_concept_extended = (
                        self.current_concept_extended.sample(frac=1).reset_index(
                            drop=True
                        )
                    )

                self.concept_list.append(
                    DataProvider(name=f"concept_{i}", generated_dataset=dataset)
                )

        elif dataset_type == "multivariate_normal":
            self.generator = SynthClassificationGenerator(
                n_features=2,
                n_outputs=1,
                n_classes=self.n_classes,
            )

            n = self.generator.size
            triangular_size = int(n * (n + 1) / 2)

            theta_values = [np.pi, np.pi / 2, np.pi / 4, 3 * np.pi / 4]

            self.mean_values = generate_mean_values(
                min_distance=3,
                n_features=self.generator.size,
                n_classes=self.generator.n_classes,
            )

            self.cov_values = [
                np.random.uniform(2, 4, triangular_size)
                for _ in range(self.generator.n_classes)
            ]

            self.generator.mean_values = self.mean_values
            self.generator.cov_values = self.cov_values

            mean_values = {}
            covariance_matrices = {}

            for i in range(n_concepts):
                mean_values[i] = {}
                covariance_matrices[i] = {}

                # perturb concept
                perturbations = create_perturbations_v2(
                    n_features=self.generator.size,
                    triangular_size=triangular_size,
                    theta_values=theta_values,
                    n_classes=self.generator.n_classes,
                    n_perturbations=4,
                )

                for class_index in range(self.generator.n_classes):
                    class_perturbation = np.random.choice(
                        perturbations[class_index], size=1
                    )[0]
                    perturbation_type = class_perturbation["type"]

                    if perturbation_type == "mean":
                        # noise in the mean vector
                        self.generator.mean_values[class_index] = (
                                self.mean_values[class_index]
                                + class_perturbation["value"]["mean"]
                        )
                        self.generator.cov_values[class_index] = self.cov_values[
                            class_index
                        ]

                        mean_values[i][class_index] = (
                                self.mean_values[class_index]
                                + class_perturbation["value"]["mean"]
                        )
                        covariance_matrices[i][
                            class_index
                        ] = self.generator.covariance_matrices[class_index]

                    elif perturbation_type == "covariance":
                        self.generator.mean_values[class_index] = self.mean_values[
                            class_index
                        ]

                        # noise in the covariance matrix
                        self.generator.cov_values[class_index] = (
                                self.cov_values[class_index]
                                + class_perturbation["value"]["cov"]
                        )

                        # rotation of the covariance matrix
                        dims = np.random.choice(
                            list(range(self.generator.size)), size=2, replace=False
                        )
                        self.generator.rotate(
                            d1=dims[0],
                            d2=dims[1],
                            theta=class_perturbation["value"]["theta"],
                            matrix_index=class_index,
                        )

                        mean_values[i][class_index] = self.mean_values[class_index]
                        covariance_matrices[i][
                            class_index
                        ] = self.generator.covariance_matrices[class_index]

                    elif perturbation_type == "combination":
                        # noise in the mean vector
                        self.generator.mean_values[class_index] = (
                                self.mean_values[class_index]
                                + class_perturbation["value"]["mean"]
                        )

                        # noise in the covariance matrix
                        self.generator.cov_values[class_index] = (
                                self.cov_values[class_index]
                                + class_perturbation["value"]["cov"]
                        )

                        # rotation of the covariance matrix
                        dims = np.random.choice(
                            list(range(self.generator.size)), size=2, replace=False
                        )
                        self.generator.rotate(
                            d1=dims[0],
                            d2=dims[1],
                            theta=class_perturbation["value"]["theta"],
                            matrix_index=class_index,
                        )

                        mean_values[i][class_index] = (
                                self.mean_values[class_index]
                                + class_perturbation["value"]["mean"]
                        )
                        covariance_matrices[i][
                            class_index
                        ] = self.generator.covariance_matrices[class_index]

                    else:
                        raise ValueError("Invalid perturbation type")

            self.concept_mapping = {}
            for i in range(n_concepts):
                self.concept_mapping["concept_" + str(i)] = {}

                alpha = 2
                # the higher the alpha, the more balanced the prior probabilities
                self.generator.prior_probs = np.random.dirichlet(
                    alpha * np.ones(self.generator.n_classes)
                )

                for class_index in range(self.generator.n_classes):
                    self.generator.mean_values[class_index] = mean_values[i][
                        class_index
                    ]
                    self.generator.covariance_matrices[
                        class_index
                    ] = covariance_matrices[i][class_index]

                    self.concept_mapping["concept_" + str(i)][
                        "class_" + str(class_index)
                        ] = multivariate_normal(
                        self.generator.mean_values[class_index],
                        self.generator.covariance_matrices[class_index],
                    )

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

                    self.test_set = self.generator.generate(self.test_set_size)
                    self.current_concept_extended = self.generator.generate(
                        n_samples=10000
                    )
                    self.current_concept_extended = (
                        self.current_concept_extended.sample(frac=1).reset_index(
                            drop=True
                        )
                    )

                self.concept_list.append(
                    DataProvider(name=f"concept_{i}", generated_dataset=dataset)
                )

        elif dataset_type == "decision_boundary":

            centers = generate_mean_values(
                min_distance=3,
                n_features=2,
                n_classes=self.n_classes,
            )
            for i in range(n_concepts):

                if i != n_concepts - 1:
                    X, _ = make_blobs(
                        n_samples=concept_size,
                        centers=centers,
                        n_features=2,
                        cluster_std=1.5,
                    )

                    classified_labels = []
                    for point in X:
                        rotated_point = rotate(point[0], point[1], 0.5 * i)
                        region_1 = custom_boundary_1(rotated_point[0], rotated_point[1]) >= 0
                        region_2 = custom_boundary_2(rotated_point[0], rotated_point[1]) >= 0

                        if region_1 and region_2:
                            classified_labels.append(0.0)
                        elif not region_1 and region_2:
                            classified_labels.append(1.0)
                        elif not region_1 and not region_2:
                            classified_labels.append(2.0)
                        else:
                            classified_labels.append(3.0)
                else:
                    concept, _ = make_blobs(
                        n_samples=last_concept_size + self.test_set_size + 10000,
                        centers=centers,
                        n_features=2,
                        cluster_std=1.5,
                    )

                    concept_labels = []
                    for point in concept:
                        rotated_point = rotate(point[0], point[1], 0.5 * i)
                        region_1 = custom_boundary_1(rotated_point[0], rotated_point[1]) >= 0
                        region_2 = custom_boundary_2(rotated_point[0], rotated_point[1]) >= 0

                        if region_1 and region_2:
                            concept_labels.append(0.0)
                        elif not region_1 and region_2:
                            concept_labels.append(1.0)
                        elif not region_1 and not region_2:
                            concept_labels.append(2.0)
                        else:
                            concept_labels.append(3.0)

                    self.rotation_angle = 0.5 * i

                    X = concept[:last_concept_size]
                    classified_labels = concept_labels[:last_concept_size]

                    self.test_set = pd.DataFrame(
                        concept[last_concept_size:last_concept_size + self.test_set_size],
                        columns=["X_0", "X_1"],
                    )
                    self.test_set["y_0"] = concept_labels[last_concept_size:last_concept_size + self.test_set_size]

                    self.current_concept_extended = pd.DataFrame(
                        concept[last_concept_size + self.test_set_size:],
                        columns=["X_0", "X_1"],
                    )
                    self.current_concept_extended["y_0"] = concept_labels[last_concept_size + self.test_set_size:]

                dataset = pd.DataFrame(X, columns=["X_0", "X_1"])
                dataset["y_0"] = classified_labels

                self.concept_list.append(
                    DataProvider(name=f"concept_{i}", generated_dataset=dataset)
                )
        elif dataset_type == "custom":

            for i in range(n_concepts):
                if i != n_concepts - 1:
                    X = generate_data(
                        data_size=concept_size,
                        x_min_max=2,
                        shape_param_1=i + 1,
                        shape_param_2=i + 1,
                        shape_param_3=i + 1,
                        shape_param_4=i + 1,
                    )
                    X["y_0"] = X["y_0"].astype("int64")
                    generated_classes = X["y_0"].unique()

                    while len(generated_classes) != 2:
                        logger.debug(
                            "Regenerating dataset, some class did not generate any sample"
                        )
                        X = generate_data(
                            data_size=concept_size,
                            x_min_max=2,
                            shape_param_1=i + 1,
                            shape_param_2=i + 1,
                            shape_param_3=i + 1,
                            shape_param_4=i + 1,
                        )
                        X["y_0"] = X["y_0"].astype("int64")
                        generated_classes = X["y_0"].unique()

                else:
                    concept = generate_data(
                        data_size=last_concept_size + self.test_set_size + 10000,
                        x_min_max=2,
                        shape_param_1=i + 1,
                        shape_param_2=i + 1,
                        shape_param_3=i + 1,
                        shape_param_4=i + 1,
                    )
                    self.shape_param = i + 1
                    concept["y_0"] = concept["y_0"].astype("int64")
                    X = concept[:last_concept_size]
                    generated_classes = X["y_0"].unique()

                    while len(generated_classes) != 2:
                        logger.debug(
                            "Regenerating dataset, some class did not generate any sample"
                        )
                        concept = generate_data(
                            data_size=last_concept_size + self.test_set_size + 10000,
                            x_min_max=2,
                            shape_param_1=i + 1,
                            shape_param_2=i + 1,
                            shape_param_3=i + 1,
                            shape_param_4=i + 1,
                        )
                        concept["y_0"] = concept["y_0"].astype("int64")
                        X = concept[:last_concept_size]
                        generated_classes = X["y_0"].unique()

                    self.test_set = concept[last_concept_size:last_concept_size + self.test_set_size]
                    self.current_concept_extended = concept[last_concept_size + self.test_set_size:]

                self.concept_list.append(
                    DataProvider(name=f"concept_{i}", generated_dataset=X)
                )

        self.metadata = {
            "past_dataset_size": (n_concepts - 1) * concept_size,
            "task": "classification",
            "type": "synth",
            "n_concepts": n_concepts,
            "concept_size": concept_size,
            "last_concept_size": last_concept_size,
            "n_samples": self.n_queries,
            "n_classes": self.n_classes,
            "test_set_size": self.test_set_size,
        }

    def run(self):
        clairvoyant = GaussianNB()

        current_concept = deepcopy(self.concept_list[-1].generated_dataset)
        X_current, y_current = self.concept_list[-1].get_split_dataset_v3()
        classes = y_current["y_0"].unique()
        clairvoyant.partial_fit(X_current, y_current.values.ravel(), classes=classes)

        test_set = deepcopy(self.test_set)
        X_test, y_test = test_set[test_set.columns[:-1]], test_set[test_set.columns[-1]]
        self.pre_AL_accuracy = clairvoyant.score(X_test, y_test)

        n_queries = self.n_queries

        while n_queries > 0:
            n_selected_samples = self.n_queries - n_queries + 1

            current_concept = pd.concat(
                (
                    current_concept,
                    self.current_concept_extended.iloc[[n_queries - 1]],
                ),
                ignore_index=True,
            )

            X_clrv, y_clrv = (
                current_concept[current_concept.columns[:-1]],
                current_concept[current_concept.columns[-1]],
            )

            clairvoyant.partial_fit(X_clrv, y_clrv.ravel())

            self.clairvoyant_accuracy[n_selected_samples] = clairvoyant.score(
                X_test, y_test
            )

            n_queries -= 1

        current_concept = pd.concat(
            (
                deepcopy(self.concept_list[-1].generated_dataset),
                self.current_concept_extended,
            ),
            ignore_index=True,
        )

        X_clrv, y_clrv = (
            current_concept[current_concept.columns[:-1]],
            current_concept[current_concept.columns[-1]],
        )

        clairvoyant.partial_fit(X_clrv, y_clrv.ravel())
        self.clairvoyant_final_accuracy = clairvoyant.score(X_test, y_test)

        for strategy in self.strategies:

            iw_handler = IWHandler(
                concept_list=deepcopy(self.concept_list),
                estimator_type=MultivariateNormalEstimator
            )

            strategy_instance: BaseStrategyV3 = strategy(
                concept_list=deepcopy(self.concept_list),
                n_samples=self.n_queries,
                current_concept_extended=self.current_concept_extended,
                concept_mapping=self.concept_mapping,
                rotation_angle=self.rotation_angle,
                shape_param=self.shape_param,
            )

            self.strategy_instances.append(strategy_instance)
            classifier = GaussianNB()

            X_current, y_current = self.concept_list[-1].get_split_dataset_v3()
            classifier.partial_fit(X_current, y_current.values.ravel(), classes=classes)

            n_queries = self.n_queries
            while n_queries > 0:
                relabeled_sample = strategy_instance.run()

                # add sample to the current concept of iw handler
                # the first computation of the ess will be done after the first query
                iw_handler.current_concept.add_samples([relabeled_sample])

                relabeled_sample = pd.DataFrame(
                    [relabeled_sample], columns=current_concept.columns
                )
                classifier.partial_fit(
                    X=relabeled_sample[relabeled_sample.columns[:-1]],
                    y=relabeled_sample[relabeled_sample.columns[-1]].ravel(),
                )

                n_selected_samples = self.n_queries - n_queries + 1
                self.AL_accuracy[
                    (strategy_instance.name, n_selected_samples)
                ] = classifier.score(X_test, y_test)

                self.ess_per_strategy[
                    (strategy_instance.name, n_selected_samples)
                ] = iw_handler.compute_effective_sample_size()

                n_queries -= 1

            self.selected_samples_per_strategy[
                strategy_instance.name
            ] = strategy_instance.all_selected_samples

    def store_results(self, experiment_index):
        simulation_results_dir = self.simulation_results_dir
        concepts_path = Path(simulation_results_dir + "/" + str(experiment_index))
        concepts_path.mkdir(parents=True, exist_ok=True)

        # Save concepts
        for concept in self.concept_list:
            concept_path = concepts_path / str(concept.name + ".csv")
            concept.generated_dataset.to_csv(concept_path, index=True)

        # Save test set
        test_set_path = Path(
            simulation_results_dir + "/" + str(experiment_index) + "/test_set.csv"
        )
        self.test_set.to_csv(test_set_path, index=False)

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

        # Save ESS
        for key, item in self.ess_per_strategy.items():
            ess_path = Path(
                simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + str(key[0])
                + "/"
                + str(key[1])
                + "_samples_ess.csv"
            )
            ess_path.parent.mkdir(parents=True, exist_ok=True)

            with open(ess_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([item])

        columns = self.concept_list[0].generated_dataset.columns
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

                # save generation metadata
            metadata_file = (
                    simulation_results_dir + "/" + str(experiment_index) + "/metadata.json"
            )
            with open(metadata_file, "w") as metadata_file:
                json.dump(self.metadata, metadata_file)

    def soft_reset(self):
        if self.generator is not None:
            self.generator.reset()
        self.metadata = None
        self.concept_mapping = None
        self.strategy_instances = []
        self.concept_list = []
        self.selected_samples_per_strategy = {}
        self.prior_probs_per_concept = []
        self.shape_param = None

        self.test_set = None

        self.AL_accuracy = {}
        self.ess_per_strategy = {}
        self.pre_AL_accuracy = 0.0


if __name__ == "__main__":
    N_EXPERIMENTS = 20

    N_SAMPLES = 200

    N_FEATURES = 2
    N_CLASSES = 5

    N_CONCEPTS = 5
    CONCEPT_SIZE = 500
    LAST_CONCEPT_SIZE = 5

    TEST_SET_SIZE = 300

    simulation = SynthClassificationSimulationV3(
        name="synth_classification_v3",
        strategies=[
            RandomSamplingStrategyV3,
            EntropyDiversitySamplingStrategy,
            EntropySamplingStrategy
        ],
        result_dir=get_root_level_dir("results"),
        n_classes=N_CLASSES,
        n_queries=N_SAMPLES,
        test_set_size=TEST_SET_SIZE,
    )

    for experiment in tqdm(range(N_EXPERIMENTS)):
        simulation.generate_dataset(
            n_concepts=N_CONCEPTS,
            concept_size=CONCEPT_SIZE,
            last_concept_size=LAST_CONCEPT_SIZE,
            dataset_type="multivariate_normal",
        )

        simulation.run()

        simulation.store_results(experiment)

        simulation.soft_reset()
