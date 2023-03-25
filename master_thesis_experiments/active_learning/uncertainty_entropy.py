import numpy as np
import pandas as pd

from master_thesis_experiments.adaptation.density_estimation import MultivariateNormalEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator

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
        self.conditional_probability_per_class = None
        self.likelihood_per_class = None
        self.new_concept_classifier = None

    def estimate_new_concept(self):
        last_concept = self.concept_list[-1]

        # fit classifier on new concept for conditional probability
        self.new_concept_classifier = LogisticRegression(multi_class='multinomial')
        X, y = last_concept.get_split_dataset()
        self.new_concept_classifier.fit(X, y)

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
        past_concepts = self.concept_list[:-1]
        for concept in past_concepts:
            X, y = concept.get_split_dataset()
            classifier = LogisticRegression(multi_class='multinomial')
            classifier.fit(X, y)
            self.classifiers[concept.name] = classifier

    def compute_label_per_concept(self):
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

    def compute_samples_uncertainty(self):

        last_concept_name = self.concept_list[-1].name
        past_concepts = self.concept_list[:-1]
        shape = (self.n_past_samples, len(self.classes))
        self.likelihood_per_class = np.ndarray(shape=shape)
        self.conditional_probability_per_class = np.ndarray(shape=shape)
        global_index = 0
        for concept in past_concepts:
            X, _ = concept.get_split_dataset()

            for index in range(len(X)):
                sample = X[index]
                for class_ in self.classes:
                    estimator = self.estimators[last_concept_name][class_]
                    self.likelihood_per_class[global_index][class_] = estimator.pdf(sample)

                self.conditional_probability_per_class[global_index] = self.new_concept_classifier.predict_proba(
                    sample.reshape(1, -1)
                )
                global_index += 1
        print(self.conditional_probability_per_class)

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
        n_samples=1
    )
    sampler.estimate_new_concept()
    sampler.build_past_classifiers()
    sampler.compute_label_per_concept()
    sampler.compute_correlation_matrix()
    sampler.compute_samples_uncertainty()
