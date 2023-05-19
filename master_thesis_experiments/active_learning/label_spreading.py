import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelSpreading
from scipy.stats.distributions import entropy

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__name__)


class LabelSpreadingStrategy(BaseStrategy):
    def __init__(
            self,
            concept_mapping,
            concept_list,
            n_samples,
            estimator_type: DensityEstimator(),
    ):
        super().__init__(concept_mapping, concept_list, n_samples, estimator_type)
        self.name = "LabelSpreading"
        self.classifiers = {}

        self.n_past_samples = None
        self.n_new_samples = None

        self.X = None
        self.y = None

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()
        self.n_past_samples = self.past_dataset.n_samples
        self.n_new_samples = self.current_concept.n_samples

        if self.X is None or self.y is None:
            self.X, self.y = self.past_dataset.get_split_dataset()

            X, y = self.current_concept.get_split_dataset()
            self.X = np.concatenate([self.X, X])
            self.y = np.concatenate([self.y, y])

    def select_samples(self):
        self.iteration += 1
        logger.debug(f"Selecting sample #{self.iteration}...")

        unlabeled_indices = np.arange(self.n_past_samples)
        y_train = np.copy(self.y)
        y_train[unlabeled_indices] = -1

        lp_model = LabelSpreading(gamma=0.25, max_iter=20)
        lp_model.fit(X=self.X, y=y_train)

        pred_entropies = entropy(lp_model.label_distributions_.T)
        pred_entropies = pd.DataFrame(pred_entropies)

        # we have to update the indexes of the uncertainty vector
        # in order to avoid using indexes of already selected samples
        updated_indexes = self.past_dataset.index.tolist()

        pred_entropies.drop(pred_entropies.tail(self.n_new_samples).index, inplace=True)
        pred_entropies.set_index(pd.Index(updated_indexes), inplace=True)

        index = pred_entropies.idxmax(axis=0)
        index = index.tolist()[0]

        sample = self.past_dataset.get_data_from_ids(index).to_numpy().ravel()
        self.selected_sample = sample

        # spostarlo alla fine per salvare i sample
        # relabelati e vedere quanti sono poi in evaluation
        self.all_selected_samples.append(self.selected_sample.tolist())
        self.relabel_samples()
        self.past_dataset.delete_sample(index)
        self.current_concept.add_samples([self.selected_sample.T])

    def run(self):
        self.initialize()
        self.select_samples()

        new_concept_list = self.concept_list
        new_concept_list[-1] = self.current_concept

        self.X = None
        self.y = None
        return new_concept_list
