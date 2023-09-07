from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from master_thesis_experiments.adaptation.density_estimation import (
    DensityEstimator,
    KernelDensityEstimator,
)

from master_thesis_experiments.simulator_toolbox.data_provider.base import DataProvider
from master_thesis_experiments.simulator_toolbox.utils import get_logger

logger = get_logger(__file__)


class BaseStrategy:
    def __init__(
            self,
            concept_mapping,
            concept_list,
            n_samples,
            prior_probs,
            estimator_type: DensityEstimator(),
            estimator_dataset: Optional[DataProvider] = None,
    ):
        self.concept_mapping = concept_mapping
        self.concept_list = concept_list
        self.n_samples = n_samples
        self.prior_probs = prior_probs
        self.estimator_type = estimator_type

        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]
        self.estimator_dataset = estimator_dataset
        self.selected_sample = None
        self.all_selected_samples = []
        self.all_relabeled_samples = []
        self.estimators = {}
        self.classes = []
        self.past_dataset = None
        self.enriched_concept = None
        self.name = ""
        self.iteration = 0

    def initialize(self):
        dataset = pd.DataFrame()
        for concept in self.past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0, ignore_index=True)

        self.past_dataset = DataProvider("past_dataset", dataset)
        output_column = dataset.columns[-1]
        self.classes = np.unique(dataset[output_column]).astype(int)

    def estimate_new_concept(self):
        logger.debug("Estimating new concept...")

        dataset: pd.DataFrame = self.current_concept.get_dataset()
        output_column = dataset.columns[-1]

        self.estimators[self.current_concept.name] = {}

        for class_ in self.classes:
            estimator = self.estimator_type(self.current_concept.name)
            filter_dataset = dataset.loc[dataset[output_column] == class_]
            filter_dataset = filter_dataset.iloc[:, :-1]
            estimator.fit(filter_dataset.to_numpy())

            self.estimators[self.current_concept.name][class_] = estimator

    @abstractmethod
    def select_samples(self):
        pass

    def relabel_samples(self):
        logger.debug("Relabeling samples...")

        # the new samples will be associated
        # the class y for which the p(x|y) is higher

        X = self.selected_sample[:-1]

        # max_posterior = -np.inf
        # for class_ in self.classes:
        #     dist = self.concept_mapping[self.current_concept.name][
        #         "class_" + str(class_)
        #     ]
        #     likelihood = dist.pdf(X)
        #     posterior = likelihood * self.prior_probs[class_]
        #     if posterior > max_posterior:
        #         max_posterior = likelihood
        #         self.selected_sample[-1] = class_

        # fare sampling per la label calcolando ogni p(x|y)
        # e facendo la normalizzazione, cosi che possa vederle
        # come delle probabilità

        pdfs = []
        for class_ in self.classes:
            dist = self.concept_mapping[self.current_concept.name][
                "class_" + str(class_)
                ]
            pdfs.append(dist.pdf(X))

        norm_pdfs = [float(i) / sum(pdfs) for i in pdfs]
        class_list = np.array(self.classes, float)
        label = np.random.choice(a=class_list, size=1, p=norm_pdfs)[0]
        self.selected_sample[-1] = label

    def add_samples_to_concept(self):
        logger.debug("Adding samples to current concept...")

        self.current_concept.add_samples([self.selected_sample])

    @abstractmethod
    def run(self):
        pass


def custom_boundary_1(x, y):
    return (x) ** 2 + y - 3


def custom_boundary_2(x, y):
    return -(x) ** 2 + y - x + 4


def rotate(x, y, theta):
    return [x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)]


class BaseStrategyV3:
    def __init__(
            self,
            concept_list,
            n_samples,
            current_concept_extended,
            concept_mapping=None,
            rotation_angle=None,
            shape_param=None,
    ):
        self.n_samples = n_samples
        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]
        self.current_concept_extended = current_concept_extended
        self.concept_mapping = concept_mapping
        self.rotation_angle = rotation_angle
        self.shape_param = shape_param

        dataset = pd.DataFrame()
        for concept in self.past_concepts:
            data = concept.get_dataset()
            dataset = pd.concat([dataset, data], axis=0, ignore_index=True)

        self.past_dataset = DataProvider("past_dataset", dataset)
        output_column = dataset.columns[-1]
        self.classes = np.unique(dataset[output_column]).astype(int)

        self.name = ""
        self.iteration = 0
        self.estimators = {}
        self.labeler = KNeighborsClassifier(n_neighbors=10)
        self.selected_sample = None
        self.all_selected_samples = []

    def estimate_new_concept(self):
        logger.debug("Estimating new concept...")

        dataset: pd.DataFrame = self.current_concept.get_dataset()
        dataset = pd.concat(
            [dataset, self.current_concept_extended], axis=0, ignore_index=True
        )
        output_column = dataset.columns[-1]

        self.estimators[self.current_concept.name] = {}

        for class_ in self.classes:
            estimator = KernelDensityEstimator(self.current_concept.name)
            filter_dataset = dataset.loc[dataset[output_column] == class_]
            filter_dataset = filter_dataset.iloc[:, :-1]
            estimator.fit(filter_dataset.to_numpy())

            self.estimators[self.current_concept.name][class_] = estimator

    def train_labeler(self):
        logger.debug("Training labeler...")

        dataset: pd.DataFrame = self.current_concept.get_dataset()
        dataset = pd.concat(
            [dataset, self.current_concept_extended], axis=0, ignore_index=True
        )

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        self.labeler.fit(X, y)

    def select_samples(self):
        pass

    def relabel_samples(self):
        logger.debug("Relabeling samples...")

        # the new samples will be associated
        # the class y for which the p(x|y) is higher

        X = self.selected_sample[:-1]

        if self.concept_mapping is not None:
            # max_posterior = -np.inf
            # for class_ in self.classes:
            #     dist = self.estimators[self.current_concept.name][
            #         class_
            #     ]
            #     likelihood = dist.pdf(X.reshape(1, -1))
            #     posterior = likelihood
            #     if posterior > max_posterior:
            #         max_posterior = likelihood
            #         self.selected_sample[-1] = class_

            # fare sampling per la label calcolando ogni p(x|y)
            # e facendo la normalizzazione, cosi che possa vederle
            # come delle probabilità
            pdfs = []
            for class_ in self.classes:
                dist = self.concept_mapping[self.current_concept.name][
                    "class_" + str(class_)
                    ]
                pdfs.append(dist.pdf(X.reshape(1, -1)))

            norm_pdfs = np.array([float(i) / sum(pdfs) for i in pdfs]).flatten()
            class_list = np.array(self.classes, float)
            label = np.random.choice(a=class_list, size=1, p=norm_pdfs)[0]
            self.selected_sample[-1] = label

        elif self.rotation_angle is not None:
            rotated_point = rotate(X[0], X[1], self.rotation_angle)
            region_1 = custom_boundary_1(rotated_point[0], rotated_point[1]) >= 0
            region_2 = custom_boundary_2(rotated_point[0], rotated_point[1]) >= 0

            if region_1 and region_2:
                self.selected_sample[-1] = 0.0
            elif not region_1 and region_2:
                self.selected_sample[-1] = 1.0
            elif not region_1 and not region_2:
                self.selected_sample[-1] = 2.0
            else:
                self.selected_sample[-1] = 3.0

        elif self.shape_param is not None:
            x_0 = X[0]
            x_1 = X[1]

            y_line = np.clip(
                (x_0 ** 3) / self.shape_param - self.shape_param * x_0 + self.shape_param * np.sin(
                    x_0 / self.shape_param), -80, 80)
            y = [x_1 > y_line]
            circle1 = np.where((x_0 - 0.5) ** 2 + (x_1 - 0.5) ** 2 < 1)[0]
            if circle1.shape[0] != 0:
                circle1 = circle1[0]
                y[circle1] = (~(np.mean(y[circle1]) > 0.5))

            circle2 = np.where((x_0 + 0.7) ** 2 + (x_1 + 1) ** 2 < 1)[0]
            if circle2.shape[0] != 0:
                circle2 = circle2[0]
                y[circle2] = (~(np.mean(y[circle2]) > 0.5))
            circle3 = np.where((x_0 - 1) ** 2 + (x_1 + 1) ** 2 < 1)[0]
            if circle3.shape[0] != 0:
                circle3 = circle3[0]
                y[circle3] = (~(np.mean(y[circle3]) > 0.5))

            circle4 = np.where((x_0 + 0.2) ** 2 + (x_1 - 0.2) ** 2 < 0.5)[0]
            if circle4.shape[0] != 0:
                circle4 = circle4[0]
                y[circle4] = (~(np.mean(y[circle4]) > 0.5))

            self.selected_sample[-1] = y[0].astype("int64")

        else:
            # utilizzo di KNN per fare il relabeling
            label = self.labeler.predict(X.reshape(1, -1))[0]
            self.selected_sample[-1] = label

    def add_samples_to_concept(self):
        logger.debug("Adding samples to current concept...")

        self.current_concept.add_samples([self.selected_sample])

    @abstractmethod
    def run(self):
        pass
