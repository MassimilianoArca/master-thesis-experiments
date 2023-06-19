import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from master_thesis_experiments.simulator_toolbox.data_provider.base import \
    DataProvider


class Simulation:
    """
    Simulation base class
    """

    def __init__(
        self, name, generator, strategies, results_dir, n_samples, estimator_type
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
        self.selected_samples_per_strategy = {}

        self.strategy_post_AL_weights = {}

        self.sim_id = self.start_time.strftime("%d-%m-%Y-%H:%M")

        path = results_dir + "/" + name + "/" + self.sim_id
        self.simulation_results_dir = path

        self.true_weights = []
        self.pre_AL_weights = []

        self.test_set = None

        self.true_p_y_given_x = pd.DataFrame()
        self.true_p_x = pd.DataFrame()

        self.pre_AL_p_y_given_x = pd.DataFrame()
        self.p_x = pd.DataFrame()

        self.p_y_given_x = {}

    def run(self):
        raise NotImplementedError

    def store_concepts(self, experiment_index):
        concepts_path = Path(self.simulation_results_dir + "/" + str(experiment_index))
        concepts_path.mkdir(parents=True, exist_ok=True)

        # Save concepts
        for concept in self.concepts:
            concept_path = concepts_path / str(concept.name + ".csv")
            concept.generated_dataset.to_csv(concept_path, index=False)

        metadata_file = (
            self.simulation_results_dir + "/" + str(experiment_index) + "/metadata.json"
        )
        with open(metadata_file, "w") as metadata_file:
            json.dump(self.metadata, metadata_file)

    def store_results(self, experiment_index):
        concepts_path = Path(self.simulation_results_dir + "/" + str(experiment_index))
        concepts_path.mkdir(parents=True, exist_ok=True)

        # Save concepts
        for concept in self.concepts:
            concept_path = concepts_path / str(concept.name + ".csv")
            concept.generated_dataset.to_csv(concept_path, index=False)

        true_p_y_given_x_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/true_conditional.csv"
        )
        self.true_p_y_given_x.to_csv(true_p_y_given_x_path, index=False)

        true_p_x_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/true_input.csv"
        )
        self.true_p_x.to_csv(true_p_x_path, index=False)

        pre_AL_p_y_given_x_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/pre_AL_conditional.csv"
        )
        self.pre_AL_p_y_given_x.to_csv(pre_AL_p_y_given_x_path, index=False)

        p_x_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/estimated_input.csv"
        )
        self.p_x.to_csv(p_x_path, index=False)

        for key, item in self.p_y_given_x.items():
            p_y_given_x_path = Path(
                self.simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + str(key[0])
                + "/"
                + str(str(key[1]) + "_samples" + ".csv")
            )
            p_y_given_x_path.parent.mkdir(parents=True, exist_ok=True)

            item.to_csv(p_y_given_x_path, index=False)

        """
        # Save weights

        self.true_weights = self.true_weights[: self.metadata["past_dataset_size"]]
        true_weights_path = Path(
            self.simulation_results_dir
            + "/"
            + str(experiment_index)
            + "/true_weights.json"
        )

        with open(true_weights_path, "w") as fp:
            json.dump(self.true_weights, fp)

        self.pre_AL_weights = self.pre_AL_weights[: self.metadata["past_dataset_size"]]
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

            self.strategy_post_AL_weights[key] = self.strategy_post_AL_weights[key][
                : self.metadata["past_dataset_size"]
            ]
            with open(post_AL_weights_path, "w") as fp:
                json.dump(self.strategy_post_AL_weights[key], fp)
            """

        columns = self.concepts[0].generated_dataset.columns
        for strategy_name, samples in self.selected_samples_per_strategy.items():
            selected_samples_path = Path(
                self.simulation_results_dir
                + "/"
                + str(experiment_index)
                + "/"
                + str(strategy_name)
                + "/"
                + "selected_samples.csv"
            )
            pd.DataFrame(samples, columns=columns).to_csv(
                selected_samples_path, index=False
            )

        # save generation metadata
        metadata_file = (
            self.simulation_results_dir + "/" + str(experiment_index) + "/metadata.json"
        )
        with open(metadata_file, "w") as metadata_file:
            json.dump(self.metadata, metadata_file)

    def soft_reset(self):
        self.generator.reset()

        self.metadata = None
        self.concept_mapping = {}
        self.strategy_instances = []
        self.concepts = []
        self.prior_probs_per_concept = []
        self.selected_samples_per_strategy = {}
        self.strategy_post_AL_weights = {}
        self.true_weights = []
        self.pre_AL_weights = []

        self.test_set = None

        self.true_p_y_given_x = pd.DataFrame()
        self.true_p_x = pd.DataFrame()

        self.pre_AL_p_y_given_x = pd.DataFrame()
        self.p_x = pd.DataFrame()

        self.p_y_given_x = {}
