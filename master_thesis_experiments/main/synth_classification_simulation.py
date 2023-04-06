from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.generator.synth_classification_generator import \
    SynthClassificationGenerator, logger
from sklearn import preprocessing
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB

from master_thesis_experiments.simulator_toolbox.simulation.base import Simulation
from master_thesis_experiments.simulator_toolbox.utils import split_dataframe_xy


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

        self.metadata = {
            "dataset_name": self.generator.name,
            "task": "classification",
            "type": "synth",
            "concept_size": concept_size,
            "last_concept_size": last_concept_size
        }

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
            prior_probs = np.random.dirichlet(np.ones(n_classes))
            self.generator.prior_probs = prior_probs
            self.prior_probs_per_concept.append(prior_probs)

            if i != n_concepts - 1:
                dataset = self.generator.generate(concept_size)
            else:
                dataset = self.generator.generate(last_concept_size)

            scaler.fit_transform(dataset)
            self.concepts.append(DataProvider('concept_' + str(i), dataset))

    def run(self):
        pass


if __name__ == '__main__':
    simulation = SynthClassificationSimulation(
        name='prova',
        generator=SynthClassificationGenerator(3, 1, 2),
        strategies=[],
        base_learners=[],
        results_dir=''
    )

    simulation.generate_dataset(3, 100, 30)

    print(simulation.concepts[0].get_dataset())
