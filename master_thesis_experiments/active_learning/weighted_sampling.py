from copy import deepcopy
from functools import reduce

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
            # gamma_handler,
            # alpha,
            # gamma,
            estimator_dataset,
            estimator_type: DensityEstimator(),
    ):
        super().__init__(
            concept_mapping=concept_mapping,
            concept_list=concept_list,
            n_samples=n_samples,
            prior_probs=prior_probs,
            estimator_dataset=estimator_dataset,
            estimator_type=estimator_type
        )
        self.name = "WeightedSampling"
        self.model = GaussianNB()
        self.weighting_handler = WeightingHandler(
            deepcopy(concept_list),
            n_samples,
            scaling_factor=1,
            similarity_measure="euclidean",
            #gamma=gamma_handler
        )
        self.weighting_handler.initialize()
        self.weights = None
        self.entropies = None

        # numerator
        self.variation_similarity_list = []

        # denominator
        self.similarity_list = []
        self.initial_random_prob = 1.0
        self.decay_rate = 0.03
        # self.alpha = alpha
        # self.gamma = gamma

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()
            # super().estimate_new_concept()

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

        # at the beginning, select samples randomly. As the number of iterations increases,
        # the probability of selecting a sample randomly decreases and the selection will be
        # based on the entropy of the samples.
        # random_sampling_prob = self.initial_random_prob * np.exp(-self.decay_rate * self.iteration)
        # if np.random.rand() < random_sampling_prob:
        #     logger.debug("Random sampling...")
        #     selected_sample_index = np.random.choice(self.past_dataset.get_dataset().index.tolist())
        #     sample = (
        #         self.past_dataset.get_data_from_ids(selected_sample_index)
        #         .to_numpy()
        #         .ravel()
        #     )
        #     self.selected_sample = sample
        #
        #     # spostarlo alla fine per salvare i sample
        #     # relabelati e vedere quanti sono poi in evaluation
        #     self.all_selected_samples.append(self.selected_sample.tolist())
        #     self.relabel_samples()
        #     self.past_dataset.delete_sample(selected_sample_index)
        #     self.current_concept.add_samples([self.selected_sample.T])
        #
        #     return selected_sample_index
        # else:
        X, _ = self.past_dataset.get_split_dataset()

        probabilities = self.model.predict_proba(X)
        entropies = pd.DataFrame(entropy(probabilities.T), columns=['entropy'])
        max_entropy = np.ones(len(self.classes)) * (1 / len(self.classes))
        entropies['entropy'] = entropies['entropy'] / entropy(max_entropy) + 1e-10  # to avoid division by zero

        indexes = self.past_dataset.get_dataset().index.tolist()
        entropies.set_index(pd.Index(indexes), inplace=True)

        # remove entry of sample selected in the previous iteration
        if self.entropies is not None:
            past_indexes = self.entropies.index.tolist()
            index_to_remove = list(set(past_indexes) - set(indexes))
            self.entropies.drop(index_to_remove, inplace=True)

        score = entropies

        # combine entropy with distance from already selected samples
        if self.all_selected_samples:
            #alpha = self.alpha

            # variazione % dell'entropia
            entropy_var_perc_matrix = np.abs((entropies - self.entropies) / self.entropies)
            entropy_var_perc_matrix.set_index(pd.Index(indexes), inplace=True)

            # similarità con sample selezionato all'istante precedente
            similarity_matrix = pd.DataFrame(pairwise.rbf_kernel(X=X, Y=self.selected_sample[:-1].reshape(1, -1), gamma=0.1), columns=['similarity'])
            similarity_matrix.set_index(pd.Index(indexes), inplace=True)

            self.variation_similarity_list.append(pd.DataFrame(entropy_var_perc_matrix['entropy'] * similarity_matrix['similarity'], columns=['var_similarity']))
            self.similarity_list.append(pd.DataFrame(similarity_matrix['similarity']))

            # update delle matrici passate prendendo solo gli indici dei samples passati non ancora scelti
            for i in range(len(self.variation_similarity_list)):
                self.variation_similarity_list[i] = self.variation_similarity_list[i].loc[indexes]
                self.similarity_list[i] = self.similarity_list[i].loc[indexes]

            numerator = reduce(lambda df1, df2: df1.add(df2, fill_value=0), self.variation_similarity_list)
            denominator = reduce(lambda df1, df2: df1.add(df2, fill_value=0), self.similarity_list)

            all_selected_samples = pd.DataFrame(self.all_selected_samples)
            all_selected_samples = all_selected_samples[all_selected_samples.columns[:-1]]

            # rbf kernel: close points have score close to 1,
            # so I subtract 1 to have close points with score close to 0.
            #
            # gamma: if high, only close together points will have a significant influence on each other,
            # while if low, points far away from each other will also have an influence on each other,
            # resulting in a smoother decision boundary.
            # so gamma scales the amount of influence two points have on each other.
            distance_matrix = 1 - pd.DataFrame(pairwise.rbf_kernel(X=X, Y=all_selected_samples, gamma=0.1))
            distance_matrix.set_index(pd.Index(indexes), inplace=True)

            # normalizzare entropia e provare la media o min
            distance_vector = distance_matrix.mean(axis=1)

            lambda_ = distance_vector / np.sqrt(self.iteration)

            # score = alpha * entropies['entropy'] + (1 - alpha) * distance_vector

            score = lambda_ * entropies['entropy'] + (1 - lambda_) * (numerator['var_similarity'] / denominator['similarity'])
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

        # at last save the entropies so that at the next iteration I
        # have the entropies of the last one
        self.entropies = entropies

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
