import numpy as np
import pandas as pd

from master_thesis_experiments.adaptation.density_estimation import MultivariateNormalEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator


class UncertaintySampling:

    def __init__(self, concept_mapping, concept_list, n_samples):
        self.concept_mapping = concept_mapping
        self.concept_list = concept_list
        self.n_samples = n_samples
        self.estimators = {}
        self.classifiers = {}
        self.label_per_concept = None

    def estimate_new_concept(self):
        last_concept = self.concept_list[-1]
        dataset: pd.DataFrame = last_concept.get_dataset()
        output_column = dataset.columns[-1]
        classes = np.unique(dataset[output_column])
        self.estimators[last_concept.name] = {}
        for class_ in classes:
            estimator = MultivariateNormalEstimator(last_concept.name)
            filter_dataset = dataset.loc[dataset[output_column] == class_]
            filter_dataset = filter_dataset.iloc[:, :-1]
            estimator.fit(
                filter_dataset.to_numpy()
            )

            self.estimators[last_concept.name][class_] = estimator

    def build_classifiers(self):
        past_concepts = self.concept_list[:-1]
        for concept in past_concepts:
            X, y = concept.get_split_dataset()
            classifier = LogisticRegression()
            classifier.fit(X, y)
            self.classifiers[concept.name] = classifier

    def compute_label_per_concept(self):
        past_concepts = self.concept_list[:-1]
        n_samples = sum(len(concept.get_dataset()) for concept in past_concepts)
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


if __name__ == '__main__':
    simulation = SynthClassificationSimulation(
        name='prova',
        generator=SynthClassificationGenerator(3, 1, 3),
        strategies=[],
        base_learners=[],
        results_dir=''
    )

    simulation.generate_dataset(3, 100, 30)

    sampler = UncertaintySampling(
        concept_mapping=simulation.concept_mapping,
        concept_list=simulation.concepts,
        n_samples=1
    )
    sampler.estimate_new_concept()
    sampler.build_classifiers()
    sampler.compute_label_per_concept()
    print(sampler.label_per_concept)
