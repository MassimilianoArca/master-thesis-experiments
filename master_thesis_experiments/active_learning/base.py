from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)


class BaseStrategy:
    def __init__(
        self,
        concept_mapping,
        concept_list,
        n_samples,
        prior_probs,
        estimator_type: DensityEstimator(),
    ):
        self.concept_mapping = concept_mapping
        self.concept_list = concept_list
        self.n_samples = n_samples
        self.prior_probs = prior_probs
        self.estimator_type = estimator_type

        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]
        self.selected_sample = None
        self.all_selected_samples = []
        self.estimators = {}
        self.classes = []
        self.past_dataset = None
        self.enriched_concept = None
        self.name = ""
        self.iteration = 0

    def initialize(self):
        dataset = pd.DataFrame()
        for concept in self.past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0, ignore_index=True)

        self.past_dataset = DataProvider("past_dataset", dataset)
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
            estimator.fit(filter_dataset.to_numpy())

            self.estimators[self.current_concept.name][class_] = estimator

    @abstractmethod
    def select_samples(self):
        pass

    def relabel_samples(self):
        logger.debug("Relabeling samples...")

        # the new samples will be associated
        # the class y for which the p(x|y) is higher

        X = self.selected_sample[:-1]

        max_posterior = -np.inf
        for class_ in self.classes:
            dist = self.concept_mapping[self.current_concept.name][
                "class_" + str(class_)
            ]
            likelihood = dist.pdf(X)
            posterior = likelihood * self.prior_probs[class_]
            if posterior > max_posterior:
                max_posterior = likelihood
                self.selected_sample[-1] = class_

        # fare sampling per la label calcolando ogni p(x|y)
        # e facendo la normalizzazione, cosi che possa vederle
        # come delle probabilità

        # pdfs = []
        # for class_ in self.classes:
        #     estimator = self.concept_mapping[self.current_concept.name][
        #         "class_" + str(class_)
        #     ]
        #     pdfs.append(estimator.pdf(X))
        #
        # norm_pdfs = [float(i) / sum(pdfs) for i in pdfs]
        # class_list = np.array(self.classes, float)
        # label = np.random.choice(a=class_list, size=1, p=norm_pdfs)[0]
        # self.selected_sample[-1] = label

    def add_samples_to_concept(self):
        logger.debug("Adding samples to current concept...")

        self.current_concept.add_samples([self.selected_sample])

    @abstractmethod
    def run(self):
        pass
