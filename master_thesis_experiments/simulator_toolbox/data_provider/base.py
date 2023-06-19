from copy import deepcopy
from os.path import exists

import pandas as pd

from master_thesis_experiments.simulator_toolbox.utils import \
    split_dataframe_xy


class DataProvider:
    """
    This class handles data
    """

    def __init__(self, name, generated_dataset):
        self.current_index = 0
        self.name = name
        self.generated_dataset = generated_dataset
        self.n_samples = generated_dataset.shape[0]
        self.enriched_dataset = None

    @property
    def index(self):
        return self.generated_dataset.index

    def get_dataset(self):
        """
        return the generated training set
        """
        return deepcopy(self.generated_dataset)

    def get_split_dataset(self):
        """
        return input reference
        """
        X, y = split_dataframe_xy(self.generated_dataset)
        return X, y

    def get_next(self):
        """
        get next sample
        """
        if not self.has_next_sample():
            raise ValueError("samples finished")
        sample = self.generated_dataset.iloc[[self.current_index]]
        self.current_index += 1
        return sample

    def has_next_sample(self):
        """
        check if there is another sample to provide
        """
        return self.current_index < len(self.generated_dataset.index)

    def get_all_data(self, available_only):
        """
        available only: get all data seen till now as dataframe
        """

        if available_only:
            return pd.concat(
                self.generated_dataset[: self.current_index + 1],
                ignore_index=True,
            )
        else:
            return pd.concat(self.generated_dataset, ignore_index=True)

    def get_data_from_ids(self, ids):
        """
        return a dataframe from ids
        """
        df = self.get_dataset()
        try:
            return df.loc[ids]
        except IndexError:
            print(f"Got indexes {ids} while dataset has indexes {df.index}")

    def add_samples(self, samples_list):
        df = pd.DataFrame(samples_list, columns=self.generated_dataset.columns)
        self.generated_dataset = pd.concat([self.generated_dataset, df], axis=0)
        self.n_samples = self.generated_dataset.shape[0]

        return deepcopy(self.generated_dataset)

    def delete_sample(self, sample_index):
        self.generated_dataset.drop(sample_index, axis=0, inplace=True)
        self.n_samples = self.generated_dataset.shape[0]
