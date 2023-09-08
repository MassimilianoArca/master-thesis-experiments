from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from master_thesis_experiments.adaptation.density_estimation import (
    DensityEstimator,
    MultivariateNormalEstimator,
)
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__name__)


class IWHandler:
    def __init__(
            self,
            concept_list,
            estimator_type: DensityEstimator(),
    ):
        self.estimator_type = estimator_type
        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]

        self.past_concepts_joints_probabilities = []
        self.current_concept_joints_probabilities = []

        self.weights_per_concept = []
        self.classes = None
        self.input_distribution_estimator = None
        self.past_input_estimators = {}
        self.past_conditional_estimators = {}
        self.current_conditional_estimator = None

        dataset: pd.DataFrame = self.current_concept.get_dataset()
        output_column = dataset.columns[-1]

        self.classes = np.unique(dataset[output_column]).astype(int)

    def estimate_past_concepts(self):
        if not bool(self.past_input_estimators):
            for concept in self.past_concepts:
                X, y = concept.get_split_dataset_v3()

                input_estimator = self.estimator_type(concept_id=concept.name)
                input_estimator.fit(X)
                conditional_estimator = GaussianNB()
                conditional_estimator.fit(X, y.to_numpy().ravel())

                self.past_input_estimators[concept.name] = input_estimator
                self.past_conditional_estimators[concept.name] = conditional_estimator

    def estimate_current_concept(self):
        X, y = self.current_concept.get_split_dataset_v3()
        if self.current_conditional_estimator is None:
            self.current_conditional_estimator = GaussianNB()
            self.current_conditional_estimator.partial_fit(
                X, y.to_numpy().ravel(), classes=self.classes
            )
        else:
            last_X = X.iloc[-1].to_frame().T
            last_y = y.iloc[-1].to_frame().T
            self.current_conditional_estimator.partial_fit(
                last_X, last_y.to_numpy().ravel()
            )

        if self.input_distribution_estimator is None:
            input_distribution_estimator = self.estimator_type(
                concept_id=self.current_concept.name
            )
            input_distribution_estimator.fit(X)
            self.input_distribution_estimator = input_distribution_estimator

    def compute_multiple_iw_weights(self):
        n_concepts = len(self.past_concepts) + 1
        n_samples = self.current_concept.get_dataset().shape[0] + np.sum(
            [concept.get_dataset().shape[0] for concept in self.past_concepts]
        )

        pdfs = np.empty(shape=(n_concepts, n_samples))
        past_concepts = deepcopy(self.past_concepts)
        past_concepts.append(deepcopy(self.current_concept))

        all_samples = np.concatenate([concept.get_dataset().to_numpy() for concept in past_concepts], axis=0)
        all_features = all_samples[:, :-1]
        all_classes = all_samples[:, -1].astype(int)

        for concept_index, concept in enumerate(past_concepts):
            if concept.name == self.current_concept.name:
                input_estimator = self.input_distribution_estimator
                conditional_estimator = self.current_conditional_estimator
            else:
                input_estimator = self.past_input_estimators[concept.name]
                conditional_estimator = self.past_conditional_estimators[concept.name]

            pdfs[concept_index] = input_estimator.pdf(all_features) * conditional_estimator.predict_proba(all_features)[
                np.arange(all_features.shape[0]), all_classes]

        target_pdfs = pdfs[-1]
        sample_per_concept_proportions = np.array(
            [concept.get_dataset().shape[0] for concept in past_concepts]) / n_samples
        behavioral_pdfs = pdfs * sample_per_concept_proportions.reshape((-1, 1))
        behavioral_pdfs = np.sum(behavioral_pdfs, axis=0)

        is_weights = target_pdfs / behavioral_pdfs
        sn_weights = is_weights / is_weights.sum()
        return sn_weights

    def compute_past_concepts_probabilities(self):
        """
        Since we compute the IW for each concept as p(x,y) / q(x,y), where q
        is the pdf of the past concept and p is the pdf of the current concept,
        we compute q(x,y) for each past concept

        """
        if not self.past_concepts_joints_probabilities:
            for concept_index, concept in enumerate(self.past_concepts):
                X, y = concept.get_split_dataset_v3()
                shape = (X.shape[0],)
                concept_joints = np.ndarray(shape=shape)

                for index in range(len(X)):
                    sample = X.loc[index].to_frame().T
                    label = y.loc[index].astype(int)

                    conditional_estimator = self.past_conditional_estimators[
                        concept.name
                    ]
                    input_estimator = self.past_input_estimators[concept.name]

                    concept_joints[index] = conditional_estimator.predict_proba(sample)[
                                                0
                                            ][label] * input_estimator.pdf(sample)
                concept_joints = pd.DataFrame(concept_joints)
                self.past_concepts_joints_probabilities.append(concept_joints)

    def compute_current_concept_probabilities(self):
        """
        Since we compute the IW for each concept as p(x,y) / q(x,y), where q
        is the pdf of the past concept and p is the pdf of the current concept,
        we compute p(x,y) for each past concept

        """
        for concept_index, concept in enumerate(self.past_concepts):
            X, y = concept.get_split_dataset_v3()
            shape = (X.shape[0],)
            concept_joints = np.ndarray(shape=shape)

            for index in range(len(X)):
                sample = X.loc[index].to_frame().T
                label = y.loc[index].astype(int)

                conditional_estimator = self.current_conditional_estimator
                input_estimator = self.input_distribution_estimator

                concept_joints[index] = conditional_estimator.predict_proba(sample)[0][
                                            label
                                        ] * input_estimator.pdf(sample)
            concept_joints = pd.DataFrame(concept_joints)
            self.current_concept_joints_probabilities.append(concept_joints)

    def compute_weights(self):
        for concept_index in range(len(self.past_concepts)):
            numerators = self.current_concept_joints_probabilities[concept_index]
            denominators = self.past_concepts_joints_probabilities[concept_index]
            weights = numerators / denominators
            self.weights_per_concept.append(weights)

        return self.weights_per_concept

    def run_weights(self):
        self.estimate_past_concepts()
        self.estimate_current_concept()

        weights_list = self.compute_multiple_iw_weights()
        return weights_list

    def compute_effective_sample_size(self):
        imp_weights = self.run_weights()
        ess = 1 / np.sum(imp_weights ** 2)
        return ess

    def soft_reset(self):
        self.weights_per_concept = []
        self.current_concept_joints_probabilities = []
