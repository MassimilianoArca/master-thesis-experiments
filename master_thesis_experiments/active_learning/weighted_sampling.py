from copy import deepcopy

import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import \
    DensityEstimator
from master_thesis_experiments.handlers.weighting_handler import \
    WeightingHandler


class WeightedSamplingStrategy(BaseStrategy):
    def __init__(
        self,
        concept_mapping,
        concept_list,
        n_samples,
        estimator_type: DensityEstimator(),
    ):
        super().__init__(concept_mapping, concept_list, n_samples, estimator_type)
        self.name = "WeightedSampling"
        self.model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        self.weighting_handler = WeightingHandler(
            deepcopy(concept_list), n_samples, scaling_factor=0.7, similarity_measure="euclidean"
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
        X, y = self.past_dataset.get_split_dataset()
        self.model.fit(X, y, sample_weight=self.weights.to_numpy().ravel())

    def select_samples(self):
        X, _ = self.past_dataset.get_split_dataset()

        probabilities = self.model.predict_proba(X)
        entropies = pd.DataFrame(entropy(probabilities.T))

        indexes = self.past_dataset.get_dataset().index.tolist()
        entropies.set_index(pd.Index(indexes), inplace=True)

        selected_sample_index = entropies.idxmax(axis=0)
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

        return deepcopy(self.current_concept.get_split_dataset())
