import random

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import MultivariateNormalEstimator, DensityEstimator
# from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator, logger


class RandomSamplingStrategy(BaseStrategy):

    def __init__(self, concept_mapping, concept_list, n_samples, estimator_type: DensityEstimator()):
        super().__init__(
            concept_mapping,
            concept_list,
            n_samples,
            estimator_type
        )
        self.name = 'RandomSampling'

    def initialize(self):
        if self.past_dataset is None:
            super().initialize()

    def select_samples(self):
        logger.debug("Selecting samples...")
        logger.info("Selecting samples...")

        past_dataset = self.past_dataset.get_dataset()

        sample_indexes = past_dataset.index.values.tolist()
        random_indexes = random.sample(sample_indexes, k=1)
        selected_samples = self.past_dataset.get_data_from_ids(random_indexes)
        self.past_dataset.delete_sample(random_indexes)
        self.selected_samples = selected_samples.to_numpy()

    def run(self):
        logger.debug("Running Random Sampling Strategy...")
        logger.info("Running Random Sampling Strategy...")
        self.initialize()
        self.estimate_new_concept()
        self.select_samples()
        self.relabel_samples()
        self.add_samples_to_concept()

        new_concepts_list = self.concept_list
        new_concepts_list[-1] = self.current_concept

        return new_concepts_list
