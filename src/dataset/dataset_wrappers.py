import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


@dataclass(frozen=True)
class DatasetWrapper:
    """Wrapper for dataset splits with metadata."""

    Xy_train: pd.DataFrame
    Xy_test: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sensitive_train: pd.Series
    sensitive_test: pd.Series
    categorical_cols: List[str]
    numerical_cols: List[str]
    target_name: str
    sensitive_name: str
    dataset_name: str


@dataclass(frozen=True)
class EncodedDatasetWrapper(DatasetWrapper):
    """
    Wrapper for processed dataset splits with encoded features.
    Inherits all attributes from DatasetWrapper and adds encoded versions
    of the features and targets.
    """

    X_enc_train: pd.DataFrame
    X_enc_test: pd.DataFrame
    Xy_enc_train: pd.DataFrame
    Xy_enc_test: pd.DataFrame
    enc_sensitive_name: str

    @classmethod
    def from_dataset(
        cls, dataset: DatasetWrapper, column_transformer: ColumnTransformer
    ) -> "EncodedDatasetWrapper":
        """Create a EncodedDatasetWrapper from a DatasetWrapper and column transformer."""

        X_enc_train, X_enc_test = cls._encode_features(dataset, column_transformer)

        Xy_enc_train, Xy_enc_test = cls._add_target_to_encoded(
            dataset, X_enc_train, X_enc_test
        )

        enc_sensitive_name = cls._get_encoded_sensitive_name(
            dataset, column_transformer
        )

        return cls(
            # Parent class attributes
            **{k: getattr(dataset, k) for k in DatasetWrapper.__dataclass_fields__},
            # New processed attributes
            X_enc_train=X_enc_train,
            X_enc_test=X_enc_test,
            Xy_enc_train=Xy_enc_train,
            Xy_enc_test=Xy_enc_test,
            enc_sensitive_name=enc_sensitive_name,
        )

    @classmethod
    def _encode_features(
        cls, dataset: DatasetWrapper, column_transformer: ColumnTransformer
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_enc_train = pd.DataFrame(
            np.array(column_transformer.fit_transform(dataset.X_train)),
            columns=column_transformer.get_feature_names_out(),
            index=dataset.X_train.index,
        )
        X_enc_test = pd.DataFrame(
            np.array(column_transformer.transform(dataset.X_test)),
            columns=column_transformer.get_feature_names_out(),
            index=dataset.X_test.index,
        )
        return X_enc_train, X_enc_test

    @classmethod
    def _add_target_to_encoded(
        cls,
        dataset: DatasetWrapper,
        X_enc_train: pd.DataFrame,
        X_enc_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add target variable to encoded features."""
        Xy_enc_train = X_enc_train.copy()
        Xy_enc_train[dataset.target_name] = dataset.y_train.astype(float)

        Xy_enc_test = X_enc_test.copy()
        Xy_enc_test[dataset.target_name] = dataset.y_test.astype(float)

        return Xy_enc_train, Xy_enc_test

    @classmethod
    def _get_encoded_sensitive_name(
        cls, dataset: DatasetWrapper, column_transformer: ColumnTransformer
    ) -> str:
        """Extract the encoded sensitive feature name."""
        enc_sensitive_names = [
            f
            for f in column_transformer.get_feature_names_out()
            if f.startswith(f"onehot__{dataset.sensitive_name}")
        ]

        if len(enc_sensitive_names) != 1:
            raise ValueError(
                f"Encoded sensitive feature {dataset.sensitive_name} was not found or is not binary."
                + f"Encoded columns: {enc_sensitive_names}"
            )

        return enc_sensitive_names[0]
