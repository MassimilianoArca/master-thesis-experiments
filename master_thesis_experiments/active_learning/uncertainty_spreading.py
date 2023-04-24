import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise

from master_thesis_experiments.active_learning.base import BaseLearner
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator, MultivariateNormalEstimator
from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import logger, \
    SynthClassificationGenerator


class UncertaintySpreadingLearner(BaseLearner):

    def __init__(self, concept_mapping, concept_list, n_samples, estimator_type: DensityEstimator()):
        super().__init__(
            concept_mapping,
            concept_list,
            n_samples,
            estimator_type
        )
        self.classifiers = {}

        self.n_past_samples = None
        self.n_new_samples = None

        self.past_samples_input_similarity = None
        self.past_samples_target_similarity = None

        self.new_samples_input_similarity = None
        self.new_samples_target_similarity = None

        self.past_samples_uncertainty = None
        self.new_samples_uncertainty = None

        self.past_samples_prediction_per_classifier = None
        self.new_samples_prediction_per_classifier = None

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()
        self.n_past_samples = self.past_dataset.n_samples
        self.n_new_samples = self.current_concept.n_samples

        self.past_samples_input_similarity = np.zeros(shape=(self.n_past_samples, self.n_past_samples))
        self.past_samples_target_similarity = np.zeros(shape=(self.n_past_samples, self.n_past_samples))

        self.new_samples_input_similarity = np.zeros(shape=(self.n_new_samples, self.n_new_samples))
        self.new_samples_target_similarity = np.zeros(shape=(self.n_new_samples, self.n_new_samples))

        self.past_samples_uncertainty = np.zeros(shape=(self.n_past_samples,))
        self.new_samples_uncertainty = np.zeros(shape=(self.n_new_samples,))

        n_concepts = len(self.past_concepts)
        shape = (self.n_new_samples, n_concepts)
        self.new_samples_prediction_per_classifier = np.ndarray(shape)

        shape = (self.n_past_samples, n_concepts)
        self.past_samples_prediction_per_classifier = np.ndarray(shape)

    def build_past_classifiers(self):
        logger.debug("Building past classifiers...")

        past_concepts = self.concept_list[:-1]
        dataset = pd.DataFrame()
        for concept in past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0)

            X, y = concept.get_split_dataset()
            classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            classifier.fit(X, y)
            self.classifiers[concept.name] = classifier

    def compute_new_samples_uncertainty(self):
        logger.debug("Computing label per concept matrix...")

        X, y = self.current_concept.get_split_dataset()
        n_total_predictions = len(self.past_concepts)
        for index in range(len(X)):
            sample = [X[index]]
            for classifier_index, classifier in enumerate(self.classifiers.values()):
                prediction = classifier.predict(sample)
                self.new_samples_prediction_per_classifier[index][classifier_index] = prediction

            true_label = y[index]
            n_right_predictions = np.count_nonzero(
                self.new_samples_prediction_per_classifier[index] == true_label
            )

            self.new_samples_uncertainty[index] = (
                    (n_total_predictions - n_right_predictions) / n_total_predictions
            )

        self.new_samples_uncertainty = pd.DataFrame(self.new_samples_uncertainty)

    def compute_past_samples_uncertainty(self):
        logger.debug("Computing samples uncertainty...")

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
                self.past_samples_prediction_per_classifier[index][classifier_index] = label
                class_counter[label] += 1

            total_count = np.sum(class_counter)
            probabilities = [count / total_count for count in class_counter]
            self.past_samples_uncertainty[index] = -sum(
                (p * math.log2(p) if p > 0 else 0) for p in probabilities
            )
        self.past_samples_uncertainty = pd.DataFrame(self.past_samples_uncertainty)

    def compute_input_similarities(self):

        # gamma to be defined
        X_past, _ = self.past_dataset.get_split_dataset()
        self.past_samples_input_similarity = pairwise.rbf_kernel(X_past, X_past)
        self.past_samples_input_similarity = pd.DataFrame(self.past_samples_input_similarity)

        # gamma to be defined
        X_new, _ = self.current_concept.get_split_dataset()
        self.new_samples_input_similarity = pairwise.rbf_kernel(X_new, X_past)
        self.new_samples_input_similarity = pd.DataFrame(self.new_samples_input_similarity).T

    def compute_target_similarities(self):

        self.past_samples_target_similarity = pairwise.pairwise_distances(
            X=self.past_samples_prediction_per_classifier, metric='jaccard'
        )
        self.past_samples_target_similarity = pd.DataFrame(self.past_samples_target_similarity)

        self.new_samples_target_similarity = pairwise.pairwise_distances(
            X=self.past_samples_prediction_per_classifier,
            Y=self.new_samples_prediction_per_classifier,
            metric='jaccard'
        )
        self.new_samples_target_similarity = pd.DataFrame(self.new_samples_target_similarity)

    def select_samples(self):
        self.compute_new_samples_uncertainty()
        self.compute_past_samples_uncertainty()
        self.compute_input_similarities()
        self.compute_target_similarities()

        joint_past_samples_similarities = self.past_samples_input_similarity * self.past_samples_target_similarity
        joint_new_samples_similarities = self.new_samples_input_similarity * self.new_samples_target_similarity

        uncertainty_vector = (
                                 joint_new_samples_similarities.dot(self.new_samples_uncertainty)
                             ) * self.past_samples_uncertainty

        uncertainty_vector = joint_past_samples_similarities.dot(uncertainty_vector)

        # we have to update the indexes of the uncertainty vector
        # in order to avoid using indexes of already selected samples
        updated_indexes = self.past_dataset.index.tolist()
        uncertainty_vector.set_index(pd.Index(updated_indexes), inplace=True)

        index = uncertainty_vector.idxmax(axis=0)
        index = index.tolist()[0]
        sample = self.past_dataset.get_data_from_ids(index).to_numpy().ravel()
        self.selected_samples.append(sample)
        self.relabel_samples()
        self.past_dataset.delete_sample(index)
        self.current_concept.add_samples([self.selected_samples[0].T])

    def run(self):
        n_samples = self.n_samples
        while n_samples > 0:
            self.initialize()
            self.build_past_classifiers()
            self.estimate_new_concept()
            self.select_samples()

            n_samples -= 1


if __name__ == '__main__':
    simulation = SynthClassificationSimulation(
        name='prova',
        generator=SynthClassificationGenerator(
            n_features=6,
            n_outputs=1,
            n_classes=4,
        ),
        strategies=[],
        base_learners=[],
        results_dir=''
    )

    simulation.generate_dataset(
        n_concepts=10,
        concept_size=60,
        last_concept_size=50
    )

    sampler = UncertaintySpreadingLearner(
        concept_mapping=simulation.concept_mapping,
        concept_list=simulation.concepts,
        n_samples=10,
        estimator_type=MultivariateNormalEstimator
    )
    sampler.run()
    print("ok")