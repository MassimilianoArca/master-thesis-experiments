import json
import os
import pickle
from copy import deepcopy
from datetime import datetime
from os.path import exists
from pathlib import Path
from typing import List

from numpy import ndarray

from master_thesis_experiments.simulator_toolbox.data_provider.base import (
    DataProvider,
)
from master_thesis_experiments.simulator_toolbox.utils import split_dataframe_xy


class Simulation:
    """
    Simulation base class
    """

    def __init__(
            self,
            name,
            generator,
            strategies,
            results_dir,
            n_samples,
            estimator_type
    ):
        self.name = name
        self.generator = generator
        self.strategies = strategies
        self.n_samples = n_samples
        self.estimator_type = estimator_type

        self.start_time = datetime.now()

        self.metadata = None
        self.strategy_instances = []
        self.concept_mapping = {}
        self.concepts: List[DataProvider] = []
        self.prior_probs_per_concept = []

        self.strategy_post_AL_weights = {}

        self.sim_id = self.start_time.strftime("%d-%m-%Y-%H:%M")

        path = results_dir + '/' + name + '/' + self.sim_id
        self.simulation_results_dir = path

        self.true_weights = []
        self.pre_AL_weights = []

    def run(self):
        raise NotImplementedError

    def store_results(self, experiment_index):

        concepts_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
        )
        concepts_path.mkdir(parents=True, exist_ok=True)

        # Save concepts
        for concept in self.concepts:
            concept_path = concepts_path / str(concept.name + ".csv")
            concept.generated_dataset.to_csv(concept_path, index=False)

        # Save weights

        self.true_weights = self.true_weights[:self.metadata["past_dataset_size"]]
        true_weights_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/true_weights.json"
        )

        with open(true_weights_path, "w") as fp:
            json.dump(self.true_weights, fp)

        self.pre_AL_weights = self.pre_AL_weights[:self.metadata["past_dataset_size"]]
        pre_AL_weights_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/pre_AL_weights.json"
        )

        with open(pre_AL_weights_path, "w") as fp:
            json.dump(self.pre_AL_weights, fp)

        for key in self.strategy_post_AL_weights:
            post_AL_weights_path = Path(
                self.simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + str(key[0])
                + "/"
                + str(str(key[1]) + "_samples" + ".json")
            )
            post_AL_weights_path.parent.mkdir(parents=True, exist_ok=True)

            self.strategy_post_AL_weights[key] = self.strategy_post_AL_weights[key][:self.metadata["past_dataset_size"]]
            with open(post_AL_weights_path, 'w') as fp:
                json.dump(self.strategy_post_AL_weights[key], fp)

        # save generation metadata
        metadata_file = (
                self.simulation_results_dir
                + "/"
                + str(experiment_index)
                + '/metadata.json'
        )
        with open(metadata_file, 'w') as metadata_file:
            json.dump(self.metadata, metadata_file)

    def soft_reset(self):
        self.generator.reset()

        # attributes for generator
        self.metadata = None
        self.concept_mapping = {}
        self.strategy_instances = []
