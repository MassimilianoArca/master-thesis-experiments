from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from master_thesis_experiments.adaptation.density_estimation import DensityEstimator, MultivariateNormalEstimator
from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator, logger


class IWHandler:

    def __init__(
            self,
            concept_mapping,
            concept_list,
            estimator_type: DensityEstimator(),
            prior_class_probabilities: List = None
    ):
        self.concept_mapping = concept_mapping
        self.concept_list = concept_list
        self.estimator_type = estimator_type
        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]

        self.joints_probabilities = []
        self.current_concept_joints_probabilities = []
        self.concepts_numerators = []

        self.true_weights_per_concept = []
        self.weights_per_concept = []
        self.classes = None
        self.prior_class_probabilities = prior_class_probabilities
        self.input_distribution_estimator = None

    def initialize(self):
        last_concept = self.concept_list[-1]
        dataset: pd.DataFrame = last_concept.get_dataset()
        output_column = dataset.columns[-1]

        self.classes = np.unique(dataset[output_column]).astype(int)
        if self.prior_class_probabilities is None:
            self.prior_class_probabilities = []
            for _ in range(len(self.past_concepts)):
                prior_class_probabilities = np.zeros(shape=(len(self.classes)))
                self.prior_class_probabilities.append(prior_class_probabilities)

    def compute_prior_probabilities(self):

        if self.prior_class_probabilities is None:
            logger.debug('Computing prior class probabilities...')

            self.prior_class_probabilities = []
            for concept in self.past_concepts:
                dataset = concept.get_dataset()
                prior_class_probabilities = np.zeros(shape=(len(self.classes)))
                for class_ in self.classes:
                    prior_class_probabilities[class_] = (
                            dataset['y_0'].value_counts()[class_] / dataset.shape[0]
                    )
                self.prior_class_probabilities.append(prior_class_probabilities)

    def compute_past_concepts_probabilities(self):
        for concept_index, concept in enumerate(self.past_concepts):
            X, y = concept.get_split_dataset()
            shape = (X.shape[0],)
            concept_joints = np.ndarray(shape=shape)

            for index in range(len(X)):
                sample = X[index]
                label = y[index].astype(int)
                class_estimator = self.concept_mapping[concept.name]['class_' + str(label)]

                concept_joints[index] = (
                        class_estimator.pdf(sample)
                        * self.prior_class_probabilities[concept_index][label]
                )
            concept_joints = pd.DataFrame(concept_joints)
            self.joints_probabilities.append(concept_joints)

    def compute_current_concept_probabilities(self):
        last_concept_name = self.concept_list[-1].name
        for concept_index, concept in enumerate(self.past_concepts):
            X, y = concept.get_split_dataset()
            shape = (X.shape[0],)
            concept_joints = np.ndarray(shape=shape)

            for index in range(len(X)):
                sample = X[index]
                label = y[index].astype(int)
                class_estimator = self.concept_mapping[last_concept_name]['class_' + str(label)]

                concept_joints[index] = (
                        class_estimator.pdf(sample)
                        * self.prior_class_probabilities[concept_index][label]
                )
            concept_joints = pd.DataFrame(concept_joints)
            self.current_concept_joints_probabilities.append(concept_joints)

    def estimate_current_concept(self):
        X, y = self.current_concept.get_split_dataset()
        conditional_probability_estimator = LogisticRegression(
            multi_class='multinomial', solver='lbfgs'
        )
        conditional_probability_estimator.fit(X, y)
        if self.input_distribution_estimator is None:
            input_distribution_estimator = self.estimator_type(concept_id=self.current_concept.name)
            input_distribution_estimator.fit(X)
            self.input_distribution_estimator = input_distribution_estimator

        return conditional_probability_estimator

    def compute_numerators(
            self,
            conditional_probability_estimator,
            input_distribution_estimator
    ):
        for concept in self.past_concepts:
            X, y = concept.get_split_dataset()
            shape = (X.shape[0],)
            numerators = np.ndarray(shape=shape)

            for index in range(len(X)):
                sample = X[index]
                label = y[index].astype(int)
                conditional_probs = conditional_probability_estimator.predict_proba(sample.reshape(1, -1))
                conditional_probs = conditional_probs.flatten()
                prob = conditional_probs[label]
                numerators[index] = prob * input_distribution_estimator.pdf(sample)

            numerators = pd.DataFrame(numerators)
            self.concepts_numerators.append(numerators)

    def compute_weights(self):

        for concept_index in range(len(self.past_concepts)):
            numerators = self.concepts_numerators[concept_index]
            denominators = self.joints_probabilities[concept_index]
            weights = numerators / denominators
            self.weights_per_concept.append(weights)

        return self.weights_per_concept

    def compute_true_weights(self):

        for concept_index in range(len(self.past_concepts)):
            numerators = self.current_concept_joints_probabilities[concept_index]
            denominators = self.joints_probabilities[concept_index]
            weights = numerators / denominators
            self.true_weights_per_concept.append(weights)

        return self.true_weights_per_concept

    def run(self):

        self.initialize()
        self.compute_prior_probabilities()
        self.compute_past_concepts_probabilities()

        conditional_probability_estimator = self.estimate_current_concept()

        self.compute_numerators(
            conditional_probability_estimator=conditional_probability_estimator,
            input_distribution_estimator=self.input_distribution_estimator
        )

        weights_list = self.compute_weights()

        current_concept_weights = np.ones(shape=self.current_concept.get_dataset().shape[0])
        current_concept_weights = pd.DataFrame(current_concept_weights)
        weights_list.append(current_concept_weights)

        imp_weights = pd.DataFrame()
        for weights in weights_list:
            imp_weights = pd.concat([imp_weights, weights], axis=0)

        imp_weights = imp_weights.to_numpy()
        return imp_weights.flatten()


if __name__ == '__main__':
    simulation = SynthClassificationSimulation(
        name='prova',
        generator=SynthClassificationGenerator(2, 1, 3),
        strategies=[],
        base_learners=[],
        results_dir=''
    )

    simulation.generate_dataset(3, 50, 30)

    iw_handler = IWHandler(
        concept_mapping=simulation.concept_mapping,
        concept_list=simulation.concepts,
        estimator_type=MultivariateNormalEstimator,
        prior_class_probabilities=simulation.prior_probs_per_concept[:-1]
    )

    iw_handler.initialize()
    iw_handler.compute_prior_probabilities()
    iw_handler.compute_past_concepts_probabilities()
    iw_handler.compute_current_concept_probabilities()

    weights_list = iw_handler.compute_true_weights()

    current_concept_weights = np.ones(shape=iw_handler.current_concept.get_dataset().shape[0])
    current_concept_weights = pd.DataFrame(current_concept_weights)
    weights_list.append(current_concept_weights)

    imp_weights = pd.DataFrame()
    for weights in weights_list:
        imp_weights = pd.concat([imp_weights, weights], axis=0)

    imp_weights = imp_weights.to_numpy()

    importance_weights = iw_handler.run()

    print(importance_weights)