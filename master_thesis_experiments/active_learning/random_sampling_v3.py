import random
from copy import deepcopy

from sklearn import preprocessing

from master_thesis_experiments.active_learning.base import BaseStrategyV3
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)

scaler = preprocessing.StandardScaler()


class RandomSamplingStrategyV3(BaseStrategyV3):
    def __init__(
        self,
        concept_list,
        n_samples,
        current_concept_extended,
        concept_mapping=None,
        rotation_angle=None,
        shape_param=None
    ):
        super().__init__(
            concept_list=concept_list,
            n_samples=n_samples,
            current_concept_extended=current_concept_extended,
            concept_mapping=concept_mapping,
            rotation_angle=rotation_angle,
            shape_param=shape_param
        )
        self.name = "RandomSamplingV3"

        # self.estimate_new_concept()
        self.train_labeler()

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

        return deepcopy(self.selected_sample)
