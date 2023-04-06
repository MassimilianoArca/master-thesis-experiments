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
            self, name, generator, strategies, results_dir, base_learners
    ):
        self.name = name
        self.generator = generator
        self.base_learners = base_learners
        self.strategies = strategies

        self.start_time = datetime.now()

        self.metadata = None
        self.concept_mapping = {}
        self.concepts: List[DataProvider] = []
        self.prior_probs_per_concept = []

        # results to be saved
        self.experiments_metadata = []

        # generated datasets
        self.strategy_logs = {}

        self.sim_id = self.start_time.strftime("%d-%m-%Y-%H:%M")

        path = results_dir + '/' + name + '/' + self.sim_id
        self.simulation_results_dir = path

        for strategy in strategies:
            self.strategy_logs[strategy.name] = []

    def init_models(self):
        """
        This method loads ml models to init the simulation
        """
        for i, concept in enumerate(self.concepts[:-1]):
            X, y = split_dataframe_xy(concept.get_dataset())

            for learner in self.base_learners:
                learner.fit(X, y)
                self.concept_mapping['concept_' + str(i)]['classifier'] = learner

    def run(self):
        raise NotImplementedError

    def store_results(self, experiment_index):

        training_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + '/training.csv'
        )
        training_path.parent.mkdir(parents=True, exist_ok=True)
        self.generated_dataset['training'].to_csv(training_path, index=False)

        # save generation metadata
        metadata_file = (
                self.simulation_results_dir
                + "/"
                + str(experiment_index)
                + '/metadata.json'
        )
        with open(metadata_file, 'w') as metadata_file:
            json.dump(self.metadata, metadata_file)

        # save generation metadata
        train_time_file = (
                self.simulation_results_dir
                + "/"
                + str(experiment_index)
                + '/train_times.json'
        )
        with open(train_time_file, 'w') as train_time_file:
            json.dump(self.train_time, train_time_file)

        # Save production with predictions
        scored_dataframe = deepcopy(self.generated_dataset['production'])
        for model_index, model_predictions in enumerate(
                self.unmanaged_predictions
        ):
            column_name = 'UNMANAGED_' + str(model_index)
            scored_dataframe[column_name] = model_predictions
        for model_index, model_predictions in enumerate(
                self.managed_predictions
        ):
            column_name = 'MANAGED_' + str(model_index)
            scored_dataframe[column_name] = model_predictions
        scored_dataframe['EXPERT'] = self.expert_predictions
        scored_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + '/scored.csv'
        )
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored_dataframe.to_csv(scored_path, index=False)

    def soft_reset(self):
        self.generator.reset()

        # attributes for generator
        self.metadata = None
        self.concept_mapping = {}

        # results to be saved
        self.experiments_metadata = []
        self.strategy_logs = {}
