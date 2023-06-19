from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise

from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider


class WeightingHandler:
    def __init__(
        self, concept_list, n_samples, scaling_factor=1, similarity_measure="euclidean"
    ):
        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]
        self.n_samples = n_samples
        self.scaling_factor = scaling_factor
        self.similarity_measure = similarity_measure
        self.past_dataset = None
        self.n_past_samples = None
        self.weights = None
        self.model = None
        self.classes = None
        self.similarities = None

    def initialize(self):
        dataset = pd.DataFrame()
        for concept in self.past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0, ignore_index=True)

        self.past_dataset = DataProvider("past_dataset", dataset)
        output_column = dataset.columns[-1]
        self.classes = np.unique(dataset[output_column])
        self.n_past_samples = self.past_dataset.n_samples

        self.weights = pd.DataFrame({"weights": np.ones(self.n_past_samples)})

        self.model = LogisticRegression(multi_class="multinomial", solver="sag")

    def compute_pre_weights(self):
        output_column = self.past_dataset.get_dataset().columns[-1]
        X_past, _ = self.past_dataset.get_split_dataset()
        X_current, _ = self.current_concept.get_split_dataset()

        X_past = pd.DataFrame(X_past)
        X_current = pd.DataFrame(X_current)

        for index, class_ in enumerate(self.classes):
            X_filtered_past = X_past.loc[
                self.past_dataset.get_dataset()[output_column] == class_
            ]
            X_filtered_past_indexes = X_filtered_past.index.values.tolist()

            X_filtered_current = X_current.loc[
                self.current_concept.get_dataset()[output_column] == class_
            ]

            similarity_matrix = pairwise.euclidean_distances(
                X_filtered_past, X_filtered_current
            )
            similarity_vector = np.sum(similarity_matrix, axis=1)

            # normalize similarity vector
            min_distance = np.min(similarity_vector)
            max_distance = np.max(similarity_vector)

            similarity_vector = (similarity_vector - min_distance) / (max_distance - min_distance)

            # weight update
            self.weights["weights"].loc[X_filtered_past_indexes] = (
                self.scaling_factor
                * self.weights["weights"].loc[X_filtered_past_indexes]
                * similarity_vector
            )

        return deepcopy(self.weights)

    def update_weights(self, selected_sample, selected_sample_index):
        sample_label = selected_sample[-1]
        sample_features = selected_sample[:-1]

        # updating datasets and weights
        self.past_dataset.delete_sample(selected_sample_index)
        self.current_concept.add_samples([selected_sample])
        self.weights.drop(selected_sample_index, axis=0, inplace=True)

        output_column = self.past_dataset.get_dataset().columns[-1]
        X_past, _ = self.past_dataset.get_split_dataset()
        X_past = pd.DataFrame(X_past)
        X_past.set_index(
            pd.Index(self.past_dataset.get_dataset().index.tolist()), inplace=True
        )

        X_filtered_past = X_past.loc[
            self.past_dataset.get_dataset()[output_column] == sample_label
        ]
        X_filtered_past_indexes = X_filtered_past.index.values.tolist()

        similarity_vector = pairwise.euclidean_distances(
            X_filtered_past, sample_features.reshape(1, -1)
        )
        similarity_vector = np.sum(similarity_vector, axis=1)

        # normalize similarity vector
        min_distance = np.min(similarity_vector)
        max_distance = np.max(similarity_vector)

        similarity_vector = (similarity_vector - min_distance) / (max_distance - min_distance)

        # weight update
        self.weights["weights"].loc[X_filtered_past_indexes] = (
            self.scaling_factor
            * self.weights["weights"].loc[X_filtered_past_indexes]
            * similarity_vector
        )

        return deepcopy(self.weights)
