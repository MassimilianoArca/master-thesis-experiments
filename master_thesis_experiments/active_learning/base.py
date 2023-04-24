from abc import abstractmethod

import numpy as np
import pandas as pd

from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import logger


class BaseLearner:

    def __init__(self, concept_mapping, concept_list, n_samples, estimator_type: DensityEstimator()):
        self.concept_mapping = concept_mapping
        self.concept_list = concept_list
        self.n_samples = n_samples
        self.estimator_type = estimator_type

        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]
        self.selected_samples = []
        self.estimators = {}
        self.classes = []
        self.past_dataset = None
        self.enriched_concept = None

    def initialize(self):
        logger.info("Initializing the learner...")

        dataset = pd.DataFrame()
        for concept in self.past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0, ignore_index=True)

        self.past_dataset = DataProvider('past_dataset', dataset)
        output_column = dataset.columns[-1]
        self.classes = np.unique(dataset[output_column]).astype(int)

    def estimate_new_concept(self):
        logger.debug("Estimating new concept...")

        dataset: pd.DataFrame = self.current_concept.get_dataset()
        output_column = dataset.columns[-1]

        self.estimators[self.current_concept.name] = {}

        for class_ in self.classes:
            estimator = self.estimator_type(self.current_concept.name)
            filter_dataset = dataset.loc[dataset[output_column] == class_]
            filter_dataset = filter_dataset.iloc[:, :-1]
            estimator.fit(
                filter_dataset.to_numpy()
            )

            self.estimators[self.current_concept.name][class_] = estimator

    @abstractmethod
    def select_samples(self):
        pass

    def relabel_samples(self):
        logger.debug("Relabeling samples...")

        # the new samples will be associated
        # the class y for which the p(x|y) is higher

        for sample in self.selected_samples:
            X = sample[:-1]
            pdf = 0
            for class_ in self.classes:
                estimator = self.estimators[self.current_concept.name][class_]
                estimator_pdf = estimator.pdf(X)
                if estimator_pdf > pdf:
                    pdf = estimator_pdf
                    sample[-1] = class_

    def add_samples_to_concept(self):
        logger.debug("Adding samples to current concept...")

        last_concept = self.current_concept
        self.enriched_concept = last_concept.add_samples(self.selected_samples)

    @abstractmethod
    def run(self):
        pass
