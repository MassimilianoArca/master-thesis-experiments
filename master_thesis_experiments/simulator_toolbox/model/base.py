from abc import abstractmethod
from copy import deepcopy

import numpy as np

from master_thesis_experiments.simulator_toolbox.utils import logtimer


class Model:
    """
    This is class wrap a ML model and provides sklearn interface
    """

    def __init__(
        self,
        name,
        ml_model,
        training_data=None,
        supports_sample_weights=True,
        version=1,
        is_trained=False,
    ):
        self.name = name
        self.ml_model = deepcopy(ml_model)
        self.version = version
        self.supports_sample_weights = supports_sample_weights
        self.is_trained = is_trained
        self.training_data = training_data

    @logtimer
    def fit(self, X, y, sample_weight=None):
        """
        fit wrapper with selection
        """
        if sample_weight is None or not self.supports_sample_weights:
            self.ml_model.fit(X, y)
        else:
            self.ml_model.fit(X, y, sample_weight)
        self.is_trained = True

    def predict_proba(self, X):
        """
        predict proba wrapper
        """
        return self.ml_model.predict_proba(X)

    def predict(self, X):
        """
        predict wrapper
        """
        return self.ml_model.predict(X)

    @abstractmethod
    def fit_support(self, X, y):
        """
        This abstract method could be used to
        fit secondary models that will support
        the primary one
        """
        pass
