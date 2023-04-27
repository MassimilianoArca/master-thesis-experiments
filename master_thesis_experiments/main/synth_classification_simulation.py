from copy import deepcopy

from master_thesis_experiments.active_learning.base import BaseStrategy
from master_thesis_experiments.adaptation.density_estimation import MultivariateNormalEstimator
from master_thesis_experiments.importance.simple import IWHandler
from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator, logger
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB

from master_thesis_experiments.simulator_toolbox.simulation.base import Simulation
from master_thesis_experiments.simulator_toolbox.utils import split_dataframe_xy, get_root_level_dir
from master_thesis_experiments.active_learning.uncertainty_spreading import UncertaintySpreadingStrategy
from master_thesis_experiments.active_learning.random_sampling import RandomSamplingStrategy
from master_thesis_experiments.active_learning.first_k_sampling import FirstKSamplingStrategy


class SynthClassificationSimulation(Simulation):

    def generate_dataset(self,
                         n_concepts,
                         concept_size,
                         last_concept_size
                         ):
        """
        This method generates the dataset
        """
        logger.debug('Generating the dataset...')

        scaler = preprocessing.StandardScaler()

        n = self.generator.size
        triangular_size = int(n * (n + 1) / 2)

        for i in range(n_concepts):
            self.concept_mapping['concept_' + str(i)] = {}

            self.generator.mean_values = [
                np.random.uniform(0, 8, self.generator.size) for _ in range(self.generator.n_classes)
            ]
            self.generator.cov_values = [
                np.random.uniform(6, 9, triangular_size) for _ in range(self.generator.n_classes)
            ]

            for j in range(self.generator.n_classes):
                self.concept_mapping['concept_' + str(i)]['class_' + str(j)] = multivariate_normal(
                    self.generator.mean_values[j], self.generator.covariance_matrices[j]
                )
            n_classes = self.generator.n_classes
            alpha = 3
            # the higher the alpha, the more balanced the prior probabilities
            prior_probs = np.random.dirichlet(alpha * np.ones(n_classes))
            self.generator.prior_probs = prior_probs
            self.prior_probs_per_concept.append(prior_probs)

            if i != n_concepts - 1:
                dataset = self.generator.generate(concept_size)
            else:
                dataset = self.generator.generate(last_concept_size)

            scaler.fit_transform(dataset)
            self.concepts.append(DataProvider('concept_' + str(i), dataset))

        self.metadata = {
            "dataset_name": self.generator.name,
            "task": "classification",
            "type": "synth",
            "n_concepts": n_concepts,
            "concept_size": concept_size,
            "last_concept_size": last_concept_size,
            "prior_probs_per_concept": self.prior_probs_per_concept,
        }

    def run(self):

        iw_handler = IWHandler(
            concept_mapping=simulation.concept_mapping,
            concept_list=simulation.concepts,
            estimator_type=simulation.estimator_type,
            prior_class_probabilities=simulation.prior_probs_per_concept[:-1]
        )

        # compute true weights
        self.true_weights = iw_handler.run_true_weights().tolist()

        # compute pre-AL weights
        self.pre_AL_weights = iw_handler.run_weights().tolist()

        for strategy in self.strategies:
            strategy_instance: BaseStrategy = strategy(
                concept_mapping=deepcopy(self.concept_mapping),
                concept_list=deepcopy(self.concepts),
                n_samples=self.n_samples,
                estimator_type=self.estimator_type
            )
            self.strategy_instances.append(strategy_instance)

            new_concept_list = strategy_instance.run()

            # compute post-AL weights
            iw_handler = IWHandler(
                concept_mapping=simulation.concept_mapping,
                concept_list=new_concept_list,
                estimator_type=simulation.estimator_type,
                prior_class_probabilities=simulation.prior_probs_per_concept[:-1]
            )

            self.strategy_post_AL_weights[strategy_instance.name] = iw_handler.run_weights().tolist()


if __name__ == '__main__':

    N_EXPERIMENTS = 10

    N_FEATURES = 4
    N_CLASSES = 3

    N_CONCEPTS = 5
    CONCEPT_SIZE = 100
    LAST_CONCEPT_SIZE = 60

    simulation = SynthClassificationSimulation(
        name='synth_classification',
        generator=SynthClassificationGenerator(
            n_features=N_FEATURES,
            n_outputs=1,
            n_classes=N_CLASSES
        ),
        strategies=[
            UncertaintySpreadingStrategy,
            RandomSamplingStrategy,
            FirstKSamplingStrategy
        ],
        results_dir=get_root_level_dir('results'),
        n_samples=10,
        estimator_type=MultivariateNormalEstimator
    )

    for experiment in tqdm(range(N_EXPERIMENTS)):
        simulation.generate_dataset(
            n_concepts=N_CONCEPTS,
            concept_size=CONCEPT_SIZE,
            last_concept_size=LAST_CONCEPT_SIZE
        )

        simulation.run()

        simulation.store_results(experiment)

        simulation.soft_reset()
