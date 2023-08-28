import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import pairwise
from sklearn.naive_bayes import GaussianNB
from scipy.stats import entropy

from master_thesis_experiments.active_learning.base import BaseStrategy, BaseStrategyV3
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)

scaler = preprocessing.StandardScaler()


class EntropyDiversitySamplingStrategy(BaseStrategyV3):
    def __init__(
        self,
        concept_list,
        n_samples,
        current_concept_extended,
        alpha,
        concept_mapping=None,
        rotation_angle=None,
    ):
        super().__init__(
            concept_list=concept_list,
            n_samples=n_samples,
            current_concept_extended=current_concept_extended,
            concept_mapping=concept_mapping,
            rotation_angle=rotation_angle,
        )
        self.name = "EntropyDiversitySampling"
        self.model = GaussianNB()
        X_current, y_current = self.current_concept.get_split_dataset_v3()
        self.model.partial_fit(
            X_current, y_current.values.ravel(), classes=self.classes.astype(float)
        )
        self.columns = self.current_concept.get_dataset().columns
        self.alpha = alpha

        self.decay_rate = 0.03
        self.initial_random_prob = 1.0

        # self.estimate_new_concept()
        self.train_labeler()

    def select_samples(self):
        self.iteration += 1
        logger.debug(f"Selecting sample #{self.iteration}...")

        # at the beginning, select samples randomly. As the number of iterations increases,
        # the probability of selecting a sample randomly decreases and the selection will be
        # based on the entropy of the samples.
        random_sampling_prob = self.initial_random_prob * np.exp(
            -self.decay_rate * self.iteration
        )
        if np.random.rand() < random_sampling_prob:
            logger.debug("Random sampling...")
            past_dataset = self.past_dataset.get_dataset()
            sample_indexes = past_dataset.index.values.tolist()
            random_index = random.sample(sample_indexes, k=1)
            selected_sample = self.past_dataset.get_data_from_ids(random_index)
            self.past_dataset.delete_sample(random_index)
            self.selected_sample = selected_sample.to_numpy().ravel()
            self.all_selected_samples.append(self.selected_sample.tolist())
            self.current_concept.add_samples([self.selected_sample.T])
        else:
            alpha = self.alpha

            X_past, _ = self.past_dataset.get_split_dataset_v3()
            past_dataset = deepcopy(self.past_dataset.get_dataset())
            sample_indexes = X_past.index.values.tolist()

            # filter past dataset to take into account only samples
            # predicted with a different label from the original one
            y_past_predicted = pd.Series(
                data=self.model.predict(X_past), index=sample_indexes
            )

            filter_condition = past_dataset["y_0"] != y_past_predicted
            past_dataset_filtered = past_dataset[filter_condition]
            X_past_filtered = past_dataset_filtered[past_dataset_filtered.columns[:-1]]
            sample_indexes_filtered = X_past_filtered.index.values.tolist()
            probabilities = self.model.predict_proba(X_past_filtered)
            entropies = pd.DataFrame(entropy(probabilities.T))

            # entropies.set_index(pd.Index(sample_indexes), inplace=True)

            X_current, _ = self.current_concept.get_split_dataset_v3()

            # rbf kernel: close points have score close to 1,
            # so I subtract 1 to have close points with score close to 0.
            #
            # gamma: if high, only close together points will have a significant influence on each other,
            # while if low, points far away from each other will also have an influence on each other,
            # resulting in a smoother decision boundary.
            # so gamma scales the amount of influence two points have on each other.
            distance_matrix = 1 - pd.DataFrame(
                pairwise.rbf_kernel(X=X_past_filtered, Y=X_current, gamma=0.8)
            )
            distance_vector = distance_matrix.mean(axis=1)

            score = alpha * entropies[0] + (1 - alpha) * distance_vector
            score = score.to_frame()

            score.set_index(pd.Index(sample_indexes_filtered), inplace=True)
            max_score_index = score.idxmax().values[0]

            selected_sample = self.past_dataset.get_data_from_ids([max_score_index])
            self.past_dataset.delete_sample(max_score_index)
            self.selected_sample = selected_sample.to_numpy().ravel()
            self.all_selected_samples.append(self.selected_sample.tolist())
            self.current_concept.add_samples([self.selected_sample.T])

    def run(self):
        logger.debug("Running Entropy Sampling Strategy...")

        self.select_samples()
        self.relabel_samples()
        self.add_samples_to_concept()

        selected_sample = pd.DataFrame([self.selected_sample], columns=self.columns)
        self.model.partial_fit(
            X=selected_sample[selected_sample.columns[:-1]],
            y=selected_sample[selected_sample.columns[-1]].ravel(),
        )

        return deepcopy(self.selected_sample)
