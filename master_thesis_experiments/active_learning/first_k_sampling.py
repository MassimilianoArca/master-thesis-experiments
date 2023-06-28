import numpy as np

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import (
    DensityEstimator,
    MultivariateNormalEstimator,
)

# from master_thesis_experiments.main.synth_classification_simulation import SynthClassificationSimulation
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import (
    SynthClassificationGenerator,
    logger,
)


class FirstKSamplingStrategy(BaseStrategy):
    def __init__(
        self,
        concept_mapping,
        concept_list,
        n_samples,
        prior_probs,
        estimator_type: DensityEstimator(),
    ):
        super().__init__(concept_mapping, concept_list, n_samples, prior_probs, estimator_type)
        self.name = "FirstKSampling"

    def select_samples(self):
        # k is the number of samples per concept to be selected
        # the first k samples of each concept will be selected

        quotient, remainder = np.divmod(self.n_samples, len(self.past_concepts))
        samples_per_concept = np.repeat(quotient, len(self.past_concepts))
        if remainder > 0:
            samples_per_concept[-1] += remainder

        for n_samples, concept in zip(samples_per_concept, self.past_concepts):
            dataset = concept.get_dataset()
            samples = dataset.head(n_samples).to_numpy()
            self.selected_samples.extend(samples)

    def run(self):
        logger.debug("Running First-K Sampling Strategy...")

        self.initialize()
        self.estimate_new_concept()
        self.select_samples()
        self.relabel_samples()
        self.add_samples_to_concept()

        new_concepts_list = self.concept_list
        new_concepts_list[-1] = self.current_concept

        return new_concepts_list
