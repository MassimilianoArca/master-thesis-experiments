import math
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import pairwise_distances

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import (
    DensityEstimator, MultivariateNormalEstimator)
from master_thesis_experiments.main.synth_classification_simulation import \
    SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import (
    SynthClassificationGenerator, logger)


class UncertaintySamplingStrategy(BaseStrategy):
    def __init__(
        self,
        concept_mapping,
        concept_list,
        n_samples,
        estimator_type: DensityEstimator(),
    ):
        super().__init__(concept_mapping, concept_list, n_samples, estimator_type)

        self.classifiers = {}
        self.label_per_concept = None
        self.correlation_matrix = None
        self.n_past_samples = None
        self.samples_uncertainty = None

    def initialize(self):
        super().initialize()

        self.n_past_samples = self.past_dataset.n_samples

        n_concepts = len(self.past_concepts)
        shape = (self.n_past_samples, n_concepts)
        self.label_per_concept = np.ndarray(shape)

        shape = (self.n_past_samples, self.n_past_samples)
        self.correlation_matrix = np.ndarray(shape, dtype=float)

        shape = (self.n_past_samples,)
        self.samples_uncertainty = np.ndarray(shape=shape)

    def build_past_classifiers(self):
        logger.debug("Building past classifiers...")

        past_concepts = self.concept_list[:-1]
        dataset = pd.DataFrame()
        for concept in past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0)

            X, y = concept.get_split_dataset()
            classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs")
            classifier.fit(X, y)
            self.classifiers[concept.name] = classifier

    def compute_label_per_concept(self):
        logger.debug("Computing label per concept matrix...")

        global_index = 0

        for concept in self.past_concepts:
            X, _ = concept.get_split_dataset()

            for index in range(len(X)):
                sample = [X[index]]
                for classifier_index, classifier in enumerate(
                    self.classifiers.values()
                ):
                    prediction = classifier.predict(sample)
                    self.label_per_concept[global_index][classifier_index] = prediction

                global_index += 1

    def compute_correlation_matrix(self):
        logger.debug("Computing correlation matrix...")

        tot_products = product(self.label_per_concept, repeat=2)

        row = 0
        column = 0
        global_index = 0
        for prod in tot_products:
            score = jaccard_score(prod[0], prod[1], average="micro")
            if global_index >= self.n_past_samples:
                global_index = 0
                column = 0
                row += 1

            self.correlation_matrix[row][column] = score
            column += 1
            global_index += 1
        self.correlation_matrix = pd.DataFrame(self.correlation_matrix)

    def compute_representativeness_matrix(self):
        logger.debug("Computing representativeness matrix...")

        X, _ = self.past_dataset.get_split_dataset()

        mahalanobis_distance = pairwise_distances(X=X, metric="mahalanobis")
        inv_distance = 1 / mahalanobis_distance
        np.fill_diagonal(inv_distance, 0)
        sum_inv_distance = np.sum(inv_distance, axis=1)
        representativeness = sum_inv_distance / np.sum(sum_inv_distance)

        self.correlation_matrix = pd.DataFrame(representativeness)

    def compute_samples_uncertainty(self):
        logger.debug("Computing samples uncertainty...")

        past_concepts_names = [past_concept.name for past_concept in self.past_concepts]
        global_index = 0

        for concept in self.past_concepts:
            X, _ = concept.get_split_dataset()

            for index in range(len(X)):
                sample = X[index]
                class_counter = np.zeros(shape=len(self.classes))

                # computing entropies
                for name_index, name in enumerate(past_concepts_names):
                    classifier = self.classifiers[name]
                    label = classifier.predict(sample.reshape(1, -1))[0].astype(int)
                    class_counter[label] += 1

                total_count = np.sum(class_counter)
                probabilities = [count / total_count for count in class_counter]
                self.samples_uncertainty[global_index] = -sum(
                    (p * math.log2(p) if p > 0 else 0) for p in probabilities
                )
                global_index += 1
        self.samples_uncertainty = pd.DataFrame(self.samples_uncertainty)

    def select_samples(self):
        logger.debug("Selecting samples...")

        assert self.n_samples > 0

        n_samples = self.n_samples
        selected_sample_index = self.samples_uncertainty[0].idxmax()
        self.samples_uncertainty.drop(selected_sample_index, inplace=True)
        self.correlation_matrix.drop(selected_sample_index, axis=0, inplace=True)
        # self.correlation_matrix.drop(selected_sample_index, axis=1, inplace=True)
        n_samples -= 1

        sample = (
            self.past_dataset.get_data_from_ids(selected_sample_index)
            .to_numpy()
            .ravel()
        )
        self.selected_samples.append(sample)

        while n_samples > 0:
            self.samples_uncertainty = self.samples_uncertainty.mul(
                self.correlation_matrix
            )

            # self.samples_uncertainty = self.samples_uncertainty.T
            selected_sample_index = self.samples_uncertainty[0].idxmax()
            sample = (
                self.past_dataset.get_data_from_ids(selected_sample_index)
                .to_numpy()
                .ravel()
            )
            self.selected_samples.append(sample)

            self.samples_uncertainty.drop(selected_sample_index, inplace=True)
            self.correlation_matrix.drop(selected_sample_index, axis=0, inplace=True)
            # self.correlation_matrix.drop(selected_sample_index, axis=1, inplace=True)
            n_samples -= 1

    def run(self):
        logger.debug("Running Uncertainty Sampling Strategy...")

        self.initialize()
        self.estimate_new_concept()
        self.build_past_classifiers()
        self.compute_representativeness_matrix()
        self.compute_samples_uncertainty()
        self.select_samples()
        self.relabel_samples()
        self.add_samples_to_concept()

        new_concepts_list = self.concept_list
        new_concepts_list[-1] = self.current_concept

        return new_concepts_list


if __name__ == "__main__":
    simulation = SynthClassificationSimulation(
        name="synth_classification",
        generator=SynthClassificationGenerator(4, 1, 3),
        strategies=[],
        results_dir="",
        n_samples=10,
        estimator_type=MultivariateNormalEstimator,
    )

    simulation.generate_dataset(10, 60, 50)

    sampler = UncertaintySamplingStrategy(
        concept_mapping=simulation.concept_mapping,
        concept_list=simulation.concepts,
        n_samples=simulation.n_samples,
        estimator_type=simulation.estimator_type,
    )
    sampler.initialize()
    sampler.compute_representativeness_matrix()
    sampler.estimate_new_concept()
    sampler.build_past_classifiers()
    # sampler.compute_label_per_concept()
    # sampler.compute_correlation_matrix()
    sampler.compute_samples_uncertainty()
    sampler.select_samples()
    sampler.relabel_samples()
    sampler.add_samples_to_concept()
