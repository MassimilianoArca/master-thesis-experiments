from array import array

import numpy as np
import pandas as pd

from master_thesis_experiments.adaptation.density_estimation import MultivariateNormalEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator, logger

from itertools import product
from sklearn.metrics import jaccard_score


class UncertaintySampling:

    def __init__(self, concept_mapping, concept_list, n_samples):
        self.concept_mapping = concept_mapping
        self.concept_list = concept_list
        self.n_samples = n_samples
        self.estimators = {}
        self.classifiers = {}
        self.label_per_concept = None
        self.correlation_matrix = None
        self.n_past_samples = None
        self.classes = []
        self.entropy_per_classifier = None
        self.likelihood_per_class = None
        self.prior_class_probabilities = None
        self.sample_probabilities = None
        self.samples_uncertainty = None
        self.past_dataset = None
        self.selected_samples = []

    def compute_prior_class_probabilities(self):

        logger.debug('Computing prior class probabilities...')
        dataset = pd.DataFrame()
        for concept in self.concept_list:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0)

        self.prior_class_probabilities = [None for _ in range(len(self.classes))]
        for class_ in self.classes:
            self.prior_class_probabilities[class_] = dataset['y_0'].value_counts()[class_] / dataset.shape[0]

    def estimate_new_concept(self):

        logger.debug("Estimating new concept...")

        last_concept = self.concept_list[-1]
        dataset: pd.DataFrame = last_concept.get_dataset()
        output_column = dataset.columns[-1]

        self.classes = np.unique(dataset[output_column]).astype(int)
        self.estimators[last_concept.name] = {}

        for class_ in self.classes:
            estimator = MultivariateNormalEstimator(last_concept.name)
            filter_dataset = dataset.loc[dataset[output_column] == class_]
            filter_dataset = filter_dataset.iloc[:, :-1]
            estimator.fit(
                filter_dataset.to_numpy()
            )

            self.estimators[last_concept.name][class_] = estimator

    def build_past_classifiers(self):

        logger.debug("Building past classifiers...")

        past_concepts = self.concept_list[:-1]
        dataset = pd.DataFrame()
        for concept in past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0)

            X, y = concept.get_split_dataset()
            classifier = LogisticRegression(multi_class='multinomial')
            classifier.fit(X, y)
            self.classifiers[concept.name] = classifier

        self.sample_probabilities = [None for _ in range(dataset.shape[0])]
        self.past_dataset = DataProvider(name='past_dataset', generated_dataset=dataset)

    def compute_label_per_concept(self):

        logger.debug("Computing label per concept...")

        past_concepts = self.concept_list[:-1]
        n_samples = sum(len(concept.get_dataset()) for concept in past_concepts)
        self.n_past_samples = n_samples
        n_concepts = len(past_concepts)
        shape = (n_samples, n_concepts)
        self.label_per_concept = np.ndarray(shape)

        global_index = 0

        for concept in past_concepts:
            X, _ = concept.get_split_dataset()

            for index in range(len(X)):
                sample = [X[index]]
                for classifier_index, classifier in enumerate(self.classifiers.values()):
                    prediction = classifier.predict(sample)
                    self.label_per_concept[global_index][classifier_index] = prediction

                global_index += 1

    def compute_correlation_matrix(self):

        logger.debug("Computing correlation matrix...")

        shape = (self.n_past_samples, self.n_past_samples)
        self.correlation_matrix = np.ndarray(shape, dtype=float)

        tot_products = product(self.label_per_concept, repeat=2)

        row = 0
        column = 0
        global_index = 0
        for prod in tot_products:
            score = jaccard_score(prod[0], prod[1], average='micro')
            if global_index >= self.n_past_samples:
                global_index = 0
                column = 0
                row += 1

            self.correlation_matrix[row][column] = score
            column += 1
            global_index += 1

    def compute_samples_probabilities(self):

        logger.debug("Computing samples uncertainty...")

        last_concept_name = self.concept_list[-1].name
        past_concepts = self.concept_list[:-1]
        past_concepts_names = [past_concept.name for past_concept in past_concepts]
        shape = (self.n_past_samples, len(self.classes))
        self.likelihood_per_class = np.ndarray(shape=shape)
        self.entropy_per_classifier = np.ndarray(shape=(self.n_past_samples, len(past_concepts)),
                                                 dtype=float)
        global_index = 0

        for concept in past_concepts:
            X, _ = concept.get_split_dataset()

            for index in range(len(X)):
                sample = X[index]
                for class_ in self.classes:
                    estimator = self.estimators[last_concept_name][class_]
                    likelihood = estimator.pdf(sample)
                    self.likelihood_per_class[global_index][class_] = likelihood

                self.sample_probabilities[global_index] = sum([
                    self.likelihood_per_class[global_index][i] *
                    self.prior_class_probabilities[i] for i in range(len(self.classes))])

                # computing entropies
                for name_index, name in enumerate(past_concepts_names):
                    classifier = self.classifiers[name]
                    conditional_prob = classifier.predict_proba(sample.reshape(1, -1))[0]
                    self.entropy_per_classifier[global_index][name_index] = - sum(
                        prob * np.log2(prob) for prob in conditional_prob
                    )

                global_index += 1

    def compute_samples_uncertainty(self):
        past_samples_size = self.entropy_per_classifier.shape[0]
        self.samples_uncertainty = np.ndarray(shape=past_samples_size)
        for index, sample in enumerate(self.entropy_per_classifier):
            self.samples_uncertainty[index] = (1 / len(sample)) * sum(entropy for entropy in sample)

        print(self.samples_uncertainty)

    def select_samples(self):
        n_samples = self.n_samples
        selected_sample_index = np.argmax(self.samples_uncertainty)
        self.samples_uncertainty[selected_sample_index] = 0
        self.correlation_matrix[selected_sample_index] = 0
        self.correlation_matrix[:, selected_sample_index] = 0
        n_samples -= 1

        self.selected_samples.append(self.past_dataset.get_data_from_ids(selected_sample_index).to_numpy())

        while n_samples > 0:
            self.samples_uncertainty = np.matmul(self.samples_uncertainty, self.correlation_matrix)
            selected_sample_index = np.argmax(self.samples_uncertainty)
            self.selected_samples.append(self.past_dataset.get_data_from_ids(selected_sample_index).to_numpy())

            self.samples_uncertainty[selected_sample_index] = 0
            self.correlation_matrix[selected_sample_index] = 0
            self.correlation_matrix[:, selected_sample_index] = 0
            n_samples -= 1

        print(self.selected_samples)


if __name__ == '__main__':
    simulation = SynthClassificationSimulation(
        name='prova',
        generator=SynthClassificationGenerator(3, 1, 3),
        strategies=[],
        base_learners=[],
        results_dir=''
    )

    simulation.generate_dataset(3, 50, 30)

    sampler = UncertaintySampling(
        concept_mapping=simulation.concept_mapping,
        concept_list=simulation.concepts,
        n_samples=3
    )
    sampler.estimate_new_concept()
    sampler.build_past_classifiers()
    sampler.compute_label_per_concept()
    sampler.compute_correlation_matrix()
    sampler.compute_prior_class_probabilities()
    sampler.compute_samples_probabilities()
    sampler.compute_samples_uncertainty()
    sampler.select_samples()
