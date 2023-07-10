import random
from copy import deepcopy

from sklearn import preprocessing

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)

scaler = preprocessing.StandardScaler()

class RandomSamplingStrategyV2(BaseStrategy):
    def __init__(
        self,
        concept_mapping,
        concept_list,
        n_samples,
        prior_probs,
        estimator_type: DensityEstimator(),
    ):
        super().__init__(concept_mapping, concept_list, n_samples, prior_probs, estimator_type)
        self.name = "RandomSamplingV2"

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()

        self.estimate_new_concept()

    def select_samples(self):
        self.iteration += 1
        logger.debug(f"Selecting sample #{self.iteration}...")

        past_dataset = self.past_dataset.get_dataset()

        sample_indexes = past_dataset.index.values.tolist()
        random_index = random.sample(sample_indexes, k=1)
        selected_sample = self.past_dataset.get_data_from_ids(random_index)
        self.past_dataset.delete_sample(random_index)
        self.selected_sample = selected_sample.to_numpy().ravel()
        self.all_selected_samples.append(self.selected_sample.tolist())

    def run(self):
        logger.debug("Running Random Sampling Strategy...")

        self.select_samples()
        self.relabel_samples()
        self.add_samples_to_concept()

        current_concept = deepcopy(self.current_concept)

        data = current_concept.generated_dataset.values
        X = data[:, :-1]
        current_concept.generated_dataset[
            current_concept.generated_dataset.columns[:-1]] = scaler.fit_transform(X)

        return current_concept.get_split_dataset()
