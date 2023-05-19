from typing import List

import numpy as np
import pandas as pd

from master_thesis_experiments.adaptation.density_estimation import DensityEstimator
from master_thesis_experiments.simulator_toolbox.model.base import Model
from master_thesis_experiments.simulator_toolbox.utils import split_dataframe_xy


class JointProbabilityHandler:
    """
    Class to handle the estimation of the joint probability
    for further analysis
    """

    def __init__(
        self,
        concept_mapping,
        concept_list,
        classifier: Model,
        estimator_type: DensityEstimator(),
        test_set: pd.DataFrame,
        prior_class_probabilities: List = None,
    ):
        self.concept_mapping = concept_mapping
        self._concept_list = concept_list
        self.classifier = classifier
        self.estimator_type = estimator_type
        self.test_set = test_set
        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]
        self.prior_class_probabilities = prior_class_probabilities

        self.classes = None
        self.input_distribution_estimator = None
        self.true_p_y_given_x = None
        self.true_p_x = None
        self.p_x = None

    @property
    def concept_list(self):
        return self._concept_list

    @concept_list.setter
    def concept_list(self, concept_list):
        self._concept_list = concept_list
        self.past_concepts = concept_list[:-1]
        self.current_concept = concept_list[-1]

    def initialize(self):
        first_concept = self._concept_list[0]
        dataset: pd.DataFrame = first_concept.get_dataset()
        output_column = dataset.columns[-1]

        self.classes = np.unique(dataset[output_column]).astype(int)
        if self.prior_class_probabilities is None:
            self.prior_class_probabilities = []
            for _ in range(len(self.past_concepts)):
                prior_class_probabilities = np.zeros(shape=(len(self.classes)))
                self.prior_class_probabilities.append(prior_class_probabilities)

        self.true_p_y_given_x = pd.DataFrame(columns=self.classes)
        self.true_p_x = pd.DataFrame(columns=self.classes)
        self.p_x = pd.DataFrame(columns=self.classes)

    def compute_true_conditional(self):
        distributions_per_class = self.concept_mapping[self.current_concept.name]
        p_x_given_y = [
            distributions_per_class["class_" + str(class_)] for class_ in self.classes
        ]

        for index, row in self.test_set.iterrows():
            denominator = sum(
                distribution.pdf(row[:-1]) * self.prior_class_probabilities[class_]
                for class_, distribution in enumerate(distributions_per_class.values())
            )

            self.true_p_y_given_x.loc[index] = (
                np.array(
                    [
                        p_x_given_y[class_].pdf(row[:-1].to_numpy())
                        * self.prior_class_probabilities[class_]
                        for class_ in self.classes
                    ]
                )
            ) / denominator

            self.true_p_x.loc[index] = denominator

        return self.true_p_y_given_x

    def estimate_conditional(self):
        current_concept = self.current_concept.generated_dataset
        X, Y = split_dataframe_xy(current_concept)
        self.classifier.fit(X, Y)
        X_test, Y_test = split_dataframe_xy(self.test_set)
        try:
            return pd.DataFrame(self.classifier.predict_proba(X_test), columns=self.classes)
        except ValueError as e:
            print(e)

    def estimate_input(self):
        current_concept = self.current_concept.generated_dataset
        X, _ = split_dataframe_xy(current_concept)
        input_distribution_estimator = self.estimator_type(
            concept_id=self.current_concept.name
        )
        input_distribution_estimator.fit(X)

        for index, row in self.test_set.iterrows():
            self.p_x.loc[index] = input_distribution_estimator.pdf(row[:-1].to_numpy())

        return self.p_x
