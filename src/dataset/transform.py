from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_transformer(
    categorical_cols: List[str], numerical_cols: List[str], drop="if_binary"
) -> ColumnTransformer:
    """Create a reusable preprocessing transformer."""
    return ColumnTransformer(
        [
            ("scale", StandardScaler(), numerical_cols),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
    )


# def inverse_transform_mixed_data(
#     transformed_data, column_transformer, cat_features, num_features
# ):
#     """
#     Inverse transform data that was previously transformed by a ColumnTransformer with
#     OneHotEncoder for categorical features and StandardScaler for numerical features.

#     Parameters:
#     -----------
#     transformed_data : pandas.DataFrame or numpy.ndarray
#         The transformed data to inverse transform
#     column_transformer : sklearn.compose.ColumnTransformer
#         The fitted ColumnTransformer that was used to transform the data
#     cat_features : list
#         The names of the categorical features in the original dataset
#     num_features : list
#         The names of the numerical features in the original dataset

#     Returns:
#     --------
#     pandas.DataFrame
#         The inverse transformed data with original feature names
#     """
#     if len(transformed_data.columns) != len(column_transformer.get_feature_names_out()):
#         raise ValueError(
#             f"Transformed data and column transformer have different number of columns. "
#             f"Transformed data has {len(transformed_data.columns)} columns, "
#             f"column transformer has {len(column_transformer.get_feature_names_out())} columns."
#         )

#     if all(transformed_data.columns != column_transformer.get_feature_names_out()):
#         raise ValueError(
#             f"Transformed data and column transformer have different columns or order of columns."
#             + f"\nTransformed data: {transformed_data.columns}\nColumn transformer: {column_transformer.get_feature_names_out()}"
#         )
#     # Convert to numpy array if it's a DataFrame
#     if isinstance(transformed_data, pd.DataFrame):
#         transformed_array = transformed_data.to_numpy()
#     else:
#         transformed_array = transformed_data

#     # Get transformers
#     transformers = column_transformer.transformers_

#     # Find the indices for numerical and categorical columns in the transformed data
#     num_cols = [
#         i
#         for i, col in enumerate(column_transformer.get_feature_names_out())
#         if col.startswith("scale__")
#     ]
#     cat_cols = [
#         i
#         for i, col in enumerate(column_transformer.get_feature_names_out())
#         if col.startswith("onehot__")
#     ]

#     # Extract data for each transformer
#     num_data = transformed_array[:, num_cols]
#     cat_data = transformed_array[:, cat_cols]

#     # Inverse transform
#     num_inverse = column_transformer.named_transformers_["scale"].inverse_transform(
#         num_data
#     )
#     cat_inverse = column_transformer.named_transformers_["onehot"].inverse_transform(
#         cat_data
#     )

#     # Create a DataFrame with the original feature names
#     result = pd.DataFrame()

#     # Add categorical features
#     for i, feature in enumerate(cat_features):
#         result[feature] = cat_inverse[:, i]

#     # Add numerical features
#     for i, feature in enumerate(num_features):
#         result[feature] = num_inverse[:, i]

#     # Reorder columns to match original order
#     result = result[cat_features + num_features]

#     return result
