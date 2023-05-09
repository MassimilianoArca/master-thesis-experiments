import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import QuantileTransformer


def make_gaussian_binary_classification(
    n_samples=1000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    random_state=42,
    n_classes=2,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
        n_classes=n_classes,
    )

    transformation = QuantileTransformer(output_distribution="normal").fit(X)

    X_normal = transformation.transform(X)

    return X_normal, y


X, y = make_gaussian_binary_classification(n_samples=10000)

if __name__ == "__main__":
    X, y = make_gaussian_binary_classification(
        n_samples=1000, n_features=3, n_informative=3, n_classes=3
    )

    columns = ["X_" + str(i) for i in range(X.shape[1])]
    columns.append("y_0")

    dataset = pd.DataFrame(np.column_stack((X, y)), columns=columns)

    print(dataset)
