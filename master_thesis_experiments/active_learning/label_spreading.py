import math
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats.distributions import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import DistanceMetric, pairwise
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelSpreading

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__name__)


def calculate_representativeness(X):
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    representativeness = 1 / (distances.mean(axis=1) + 1e-6)
    return representativeness


def calculate_diversity(X):
    dist_metric = DistanceMetric.get_metric("euclidean")
    pairwise_distances = dist_metric.pairwise(X)
    diversity = 1 / (pairwise_distances.mean(axis=1) + 1e-6)
    return diversity


class LabelSpreadingStrategy(BaseStrategy):
    def __init__(
        self,
        concept_mapping,
        concept_list,
        n_samples,
        prior_probs,
        estimator_type: DensityEstimator(),
    ):
        super().__init__(
            concept_mapping, concept_list, n_samples, prior_probs, estimator_type
        )
        self.name = "LabelSpreading"
        self.classifiers = {}

        self.n_past_samples = None
        self.n_new_samples = None

        self.X = None
        self.y = None

        self.past_samples_uncertainty = None
        self.new_samples_uncertainty = None

        self.past_samples_prediction_per_classifier = None
        self.new_samples_prediction_per_classifier = None

        self.past_samples_target_similarity = None
        self.new_samples_target_similarity = None

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()

        if self.X is None or self.y is None:
            self.X, self.y = self.past_dataset.get_split_dataset()

            X, y = self.current_concept.get_split_dataset()
            self.X = np.concatenate([self.X, X])
            self.y = np.concatenate([self.y, y])

        self.n_past_samples = self.past_dataset.n_samples
        self.n_new_samples = self.current_concept.n_samples

        self.past_samples_target_similarity = np.zeros(
            shape=(self.n_past_samples, self.n_past_samples)
        )
        self.new_samples_target_similarity = np.zeros(
            shape=(self.n_past_samples, self.n_new_samples)
        )

        self.past_samples_uncertainty = np.zeros(shape=(self.n_past_samples,))
        self.new_samples_uncertainty = np.zeros(shape=(self.n_new_samples,))

        n_concepts = len(self.past_concepts)
        shape = (self.n_new_samples, n_concepts)
        self.new_samples_prediction_per_classifier = np.ndarray(shape)

        shape = (self.n_past_samples, n_concepts)
        self.past_samples_prediction_per_classifier = np.ndarray(shape)

        if self.enriched_concept is None:
            self.enriched_concept = deepcopy(self.current_concept)

    def build_past_classifiers(self):
        logger.debug("Building past classifiers...")

        past_concepts = self.concept_list[:-1]
        dataset = pd.DataFrame()
        for concept in past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0)

            X, y = concept.get_split_dataset()
            classifier = LogisticRegression(multi_class="multinomial")
            classifier.fit(X, y)
            self.classifiers[concept.name] = classifier

    def compute_new_samples_uncertainty(self):
        logger.debug("Computing new samples uncertainty...")

        X, y = self.current_concept.get_split_dataset()
        n_total_predictions = len(self.past_concepts)
        for index in range(len(X)):
            sample = [X[index]]
            for classifier_index, classifier in enumerate(self.classifiers.values()):
                prediction = classifier.predict(sample)
                self.new_samples_prediction_per_classifier[index][
                    classifier_index
                ] = prediction

            true_label = y[index]
            n_right_predictions = np.count_nonzero(
                self.new_samples_prediction_per_classifier[index] == true_label
            )

            self.new_samples_uncertainty[index] = (
                n_total_predictions - n_right_predictions
            ) / n_total_predictions

        self.new_samples_uncertainty = pd.DataFrame(self.new_samples_uncertainty)

        # set to 0 the uncertainty of the new added samples,
        # since they are selected samples from the past
        new_added_samples_size = (
            self.enriched_concept.n_samples - self.current_concept.n_samples
        )
        self.new_samples_uncertainty.loc[
            self.new_samples_uncertainty.tail(new_added_samples_size).index
        ] = 0

    def compute_past_samples_uncertainty(self):
        logger.debug("Computing past samples uncertainty...")

        # computing past samples uncertainty
        past_concepts_names = [past_concept.name for past_concept in self.past_concepts]

        X, _ = self.past_dataset.get_split_dataset()

        for index in range(len(X)):
            sample = X[index]
            class_counter = np.zeros(shape=len(self.classes))

            # computing entropies
            for classifier_index, name in enumerate(past_concepts_names):
                classifier = self.classifiers[name]
                label = classifier.predict(sample.reshape(1, -1))[0].astype(int)
                self.past_samples_prediction_per_classifier[index][
                    classifier_index
                ] = label
                class_counter[label] += 1

            total_count = np.sum(class_counter)
            probabilities = [count / total_count for count in class_counter]
            self.past_samples_uncertainty[index] = -sum(
                (p * math.log2(p) if p > 0 else 0) for p in probabilities
            )
        self.past_samples_uncertainty = pd.DataFrame(self.past_samples_uncertainty)

    def compute_target_similarities(self):
        logger.debug("Computing target similarities...")

        self.past_samples_target_similarity = pairwise.pairwise_distances(
            X=self.past_samples_prediction_per_classifier, metric="jaccard"
        )
        self.past_samples_target_similarity = pd.DataFrame(
            self.past_samples_target_similarity
        )

        self.new_samples_target_similarity = pairwise.pairwise_distances(
            X=self.past_samples_prediction_per_classifier,
            Y=self.new_samples_prediction_per_classifier,
            metric="jaccard",
        )
        self.new_samples_target_similarity = pd.DataFrame(
            self.new_samples_target_similarity
        )

    def select_samples(self):
        self.iteration += 1
        logger.debug(f"Selecting sample #{self.iteration}...")

        unlabeled_indices = np.arange(self.n_past_samples)
        y_train = np.copy(self.y)
        y_train[unlabeled_indices] = -1

        lp_model = LabelSpreading(gamma=0.25, max_iter=20)
        lp_model.fit(X=self.X, y=y_train)

        pred_entropies = entropy(lp_model.label_distributions_.T)
        pred_entropies = pd.DataFrame(pred_entropies)
        pred_entropies.drop(pred_entropies.tail(self.n_new_samples).index, inplace=True)

        representativeness = pd.DataFrame(calculate_representativeness(self.X))
        representativeness.drop(
            representativeness.tail(self.n_new_samples).index, inplace=True
        )

        diversity = pd.DataFrame(calculate_diversity(self.X))
        diversity.drop(diversity.tail(self.n_new_samples).index, inplace=True)

        # new_samples_uncertainty = self.new_samples_target_similarity @ self.new_samples_uncertainty

        score = 0.7 * pred_entropies + 0.5 * representativeness + 2 * diversity

        # we have to update the indexes of the uncertainty vector
        # in order to avoid using indexes of already selected samples
        updated_indexes = self.past_dataset.index.tolist()

        score.set_index(pd.Index(updated_indexes), inplace=True)

        index = score.idxmax(axis=0)
        index = index.tolist()[0]

        sample = self.past_dataset.get_data_from_ids(index).to_numpy().ravel()
        self.selected_sample = sample

        # spostarlo alla fine per salvare i sample
        # relabelati e vedere quanti sono poi in evaluation
        self.all_selected_samples.append(self.selected_sample.tolist())
        self.relabel_samples()
        self.past_dataset.delete_sample(index)
        self.enriched_concept.add_samples([self.selected_sample.T])

    def run(self):
        self.initialize()
        self.build_past_classifiers()
        # self.compute_new_samples_uncertainty()
        # self.compute_past_samples_uncertainty()
        # self.compute_target_similarities()
        self.select_samples()

        new_concept_list = self.concept_list
        new_concept_list[-1] = self.enriched_concept

        self.X = None
        self.y = None
        return new_concept_list
