from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import pairwise

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.handlers.weighting_handler import WeightingHandler
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)

scaler = preprocessing.StandardScaler()


class WeightedSamplingStrategy(BaseStrategy):
    def __init__(
            self,
            concept_mapping,
            concept_list,
            n_samples,
            prior_probs,
            estimator_type: DensityEstimator(),
    ):
        super().__init__(concept_mapping, concept_list, n_samples, prior_probs, estimator_type)
        self.name = "WeightedSampling"
        self.model = GaussianNB()
        self.weighting_handler = WeightingHandler(
            deepcopy(concept_list),
            n_samples,
            scaling_factor=1,
            similarity_measure="euclidean",
        )
        self.weighting_handler.initialize()
        self.weights = None

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()

        self.compute_pre_weights()

    def compute_pre_weights(self):
        self.weights = self.weighting_handler.compute_pre_weights()

    def train_model(self):
        logger.debug("Training model...")

        past_dataset = deepcopy(self.past_dataset)
        data = past_dataset.generated_dataset.values
        X = data[:, :-1]
        past_dataset.generated_dataset[past_dataset.generated_dataset.columns[:-1]] = scaler.fit_transform(X)
        X_past, y_past = past_dataset.get_split_dataset()

        current_concept = deepcopy(self.current_concept)
        data = current_concept.generated_dataset.values
        X = data[:, :-1]
        current_concept.generated_dataset[current_concept.generated_dataset.columns[:-1]] = scaler.fit_transform(X)
        X_current, y_current = current_concept.get_split_dataset()

        X = np.concatenate((X_past, X_current), axis=0)
        y = np.concatenate((y_past, y_current), axis=0)

        X_current_shape = X_current.shape[0]
        sample_weight = pd.concat(
            [self.weights, pd.DataFrame({"weights": [1] * X_current_shape})],
            axis=0,
            ignore_index=True,
        )

        self.model.fit(X, y, sample_weight=sample_weight.to_numpy().ravel())

    def select_samples(self):
        self.iteration += 1
        logger.debug(f"Selecting sample #{self.iteration}...")

        X, _ = self.past_dataset.get_split_dataset()

        probabilities = self.model.predict_proba(X)
        entropies = pd.DataFrame(entropy(probabilities.T), columns=['entropy'])
        max_entropy = np.ones(len(self.classes)) * (1 / len(self.classes))
        entropies['entropy'] = entropies['entropy'] / entropy(max_entropy)

        indexes = self.past_dataset.get_dataset().index.tolist()

        score = entropies

        # combine entropy with distance from already selected samples
        if self.all_selected_samples:
            alpha = 0.4

            all_selected_samples = pd.DataFrame(self.all_selected_samples)
            all_selected_samples = all_selected_samples[all_selected_samples.columns[:-1]]

            # rbf kernel: close points have score close to 1,
            # so I subtract 1 to have close points with score close to 0
            distance_matrix = 1 - pd.DataFrame(pairwise.rbf_kernel(X=X, Y=all_selected_samples, gamma=0.1))

            # normalizzare entropia e provare la media o min
            distance_vector = distance_matrix.min(axis=1)

            score = alpha * entropies['entropy'] + (1 - alpha) * distance_vector
            score = score.to_frame()

        score.set_index(pd.Index(indexes), inplace=True)
        selected_sample_index = score.idxmax(axis=0)
        selected_sample_index = selected_sample_index.tolist()[0]

        sample = (
            self.past_dataset.get_data_from_ids(selected_sample_index)
            .to_numpy()
            .ravel()
        )
        self.selected_sample = sample

        # spostarlo alla fine per salvare i sample
        # relabelati e vedere quanti sono poi in evaluation
        self.all_selected_samples.append(self.selected_sample.tolist())
        self.relabel_samples()
        self.past_dataset.delete_sample(selected_sample_index)
        self.current_concept.add_samples([self.selected_sample.T])

        return selected_sample_index

    def run(self):
        self.train_model()
        selected_sample_index = self.select_samples()

        self.weights = self.weighting_handler.update_weights(
            self.selected_sample, selected_sample_index
        )

        current_concept = deepcopy(self.current_concept)

        data = current_concept.generated_dataset.values
        X = data[:, :-1]
        current_concept.generated_dataset[
            current_concept.generated_dataset.columns[:-1]] = scaler.fit_transform(X)

        return current_concept.get_split_dataset()
