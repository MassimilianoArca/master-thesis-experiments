import random
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import entropy

from master_thesis_experiments.active_learning.base import BaseStrategy, BaseStrategyV3
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)

scaler = preprocessing.StandardScaler()


class EntropySamplingStrategy(BaseStrategyV3):
    def __init__(
        self,
        concept_list,
        n_samples,
        current_concept_extended,
        concept_mapping=None,
        rotation_angle=None,
        shape_param=None,
    ):
        super().__init__(
            concept_list=concept_list,
            n_samples=n_samples,
            current_concept_extended=current_concept_extended,
            concept_mapping=concept_mapping,
            rotation_angle=rotation_angle,
            shape_param=shape_param,
        )
        self.name = "EntropySampling"
        self.model = SGDClassifier(loss="log_loss", random_state=42)
        X_current, y_current = self.current_concept.get_split_dataset_v3()
        self.model.partial_fit(
            X_current, y_current.values.ravel(), classes=self.classes.astype(float)
        )
        self.columns = self.current_concept.get_dataset().columns

        # self.estimate_new_concept()
        self.train_labeler()

    def select_samples(self):
        self.iteration += 1
        logger.debug(f"Selecting sample #{self.iteration}...")

        X_past, y_past = self.past_dataset.get_split_dataset_v3()
        probabilities = self.model.predict_proba(X_past)
        entropies = pd.DataFrame(entropy(probabilities.T))

        # # normalize entropy
        # max_entropy = np.ones(len(self.classes)) * (1 / len(self.classes))
        # entropies = entropies / entropies.sum(axis=0)
        # if entropies.isnull().any().any():
        #     print("Entropy is null")

        sample_indexes = X_past.index.values.tolist()
        entropies.set_index(pd.Index(sample_indexes), inplace=True)

        max_entropy_index = entropies.idxmax().values[0]
        print(f"Max entropy index: {max_entropy_index}")
        selected_sample = self.past_dataset.get_data_from_ids([max_entropy_index])

        self.past_dataset.delete_sample(max_entropy_index)
        self.selected_sample = selected_sample.to_numpy().ravel()
        self.all_selected_samples.append(self.selected_sample.tolist())

    def run(self):
        logger.debug("Running Entropy Sampling Strategy...")

        self.select_samples()
        self.relabel_samples()
        self.add_samples_to_concept()

        selected_sample = pd.DataFrame([self.selected_sample], columns=self.columns)
        self.model.partial_fit(
            X=selected_sample[selected_sample.columns[:-1]],
            y=selected_sample[selected_sample.columns[-1]].ravel(),
        )

        return deepcopy(self.selected_sample)
