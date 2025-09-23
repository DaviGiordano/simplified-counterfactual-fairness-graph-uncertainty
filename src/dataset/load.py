from pathlib import Path

import pandas as pd
from fairlearn.datasets import (
    fetch_adult,
    fetch_bank_marketing,
    fetch_credit_card,
    fetch_diabetes_hospital,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.dataset.dataset_wrappers import DatasetWrapper, EncodedDatasetWrapper
from src.dataset.metadata import apply_metadata_conversions
from src.dataset.transform import get_column_transformer
from src.utils import find_root


def load_dataset_col_trf(name) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
    if name == "adult":
        return load_adult()
    if name == "adult_small":
        return load_adult_small()
    if name == "compas":
        return load_compas()
    if name == "law_school":
        return load_law_school()
        # if name == "bank_marketing":
        #     return load_bank_marketing()
        # if name == "diabetes_hospital":
        #     return load_diabetes_hospital()
        # if name == "credit_card":
        #     return load_credit_card()
    if name == "synthetic":
        raw_data = load_synthetic()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    col_trf = get_column_transformer(raw_data.categorical_cols, raw_data.numerical_cols)
    enc_dataset = EncodedDatasetWrapper.from_dataset(raw_data, col_trf)
    return enc_dataset, col_trf


def load_synthetic() -> DatasetWrapper:
    BASE_PATH = find_root()
    SYNTHETIC_TRAIN_FPATH = BASE_PATH / "data/synthetic/train.csv"
    SYNTHETIC_TEST_FPATH = BASE_PATH / "data/synthetic/test.csv"
    TETRAD_METADATA_FPATH = BASE_PATH / "data/synthetic/metadata.json"

    df_train = apply_metadata_conversions(
        pd.read_csv(SYNTHETIC_TRAIN_FPATH),
        TETRAD_METADATA_FPATH,
    )
    df_test = apply_metadata_conversions(
        pd.read_csv(SYNTHETIC_TEST_FPATH),
        TETRAD_METADATA_FPATH,
    )

    SENSITIVE_NAME = "A"
    TARGET_NAME = "Y"
    CATEGORICAL_COLS = ["A"]
    NUMERICAL_COLS = ["X1", "X2"]
    features = NUMERICAL_COLS + [SENSITIVE_NAME]

    return DatasetWrapper(
        Xy_train=df_train,
        Xy_test=df_test,
        X_train=df_train[features],
        X_test=df_test[features],
        y_train=df_train[TARGET_NAME],
        y_test=df_test[TARGET_NAME],
        sensitive_train=df_train[SENSITIVE_NAME],
        sensitive_test=df_test[SENSITIVE_NAME],
        categorical_cols=CATEGORICAL_COLS,
        numerical_cols=NUMERICAL_COLS,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="synthetic",
    )


def load_adult(
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
    """Load and split the Adult dataset, wrapped in a Dataset dataclass."""

    SENSITIVE_NAME = "sex"
    TARGET_NAME = "income"
    BASE_PATH = find_root()

    TETRAD_METADATA_FPATH = BASE_PATH / "data/adult/metadata.json"

    adult = fetch_adult(as_frame=True)
    X = adult.data.drop(columns="fnlwgt")
    y = adult.target.map({"<=50K": 0, ">50K": 1})

    categorical_cols = X.select_dtypes(include="category").columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X[categorical_cols] = X[categorical_cols].astype("object")
    X.fillna("Missing", inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train = apply_metadata_conversions(
        X_train,
        TETRAD_METADATA_FPATH,
    )

    sensitive_train = X_train[SENSITIVE_NAME]
    sensitive_test = X_test[SENSITIVE_NAME]

    raw_data = DatasetWrapper(
        Xy_train=pd.concat([X_train, y_train], axis=1),
        Xy_test=pd.concat([X_test, y_test], axis=1),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="adult",
    )
    col_trf = get_column_transformer(raw_data.categorical_cols, raw_data.numerical_cols)

    enc_data = EncodedDatasetWrapper.from_dataset(
        raw_data,
        col_trf,
    )
    return enc_data, col_trf


def load_adult_small(
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
    """Load and split the Adult dataset, wrapped in a Dataset dataclass."""

    SENSITIVE_NAME = "sex"
    TARGET_NAME = "income"
    BASE_PATH = find_root()

    TETRAD_METADATA_FPATH = BASE_PATH / "data/adult/metadata.json"

    adult = fetch_adult(as_frame=True)
    X = adult.data.drop(columns="fnlwgt")
    y = adult.target.map({"<=50K": 0, ">50K": 1})

    # categorical_cols = X.select_dtypes(include="category").columns.tolist()
    # numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols = ["age", "education-num", "hours-per-week"]
    categorical_cols = ["occupation", "sex"]
    X[categorical_cols] = X[categorical_cols].astype("object")
    X[categorical_cols] = X[categorical_cols].fillna("Missing")
    X = X[categorical_cols + numerical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train = apply_metadata_conversions(
        X_train,
        TETRAD_METADATA_FPATH,
    )

    sensitive_train = X_train[SENSITIVE_NAME]
    sensitive_test = X_test[SENSITIVE_NAME]

    raw_data = DatasetWrapper(
        Xy_train=pd.concat([X_train, y_train], axis=1),
        Xy_test=pd.concat([X_test, y_test], axis=1),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="adult",
    )
    col_trf = get_column_transformer(raw_data.categorical_cols, raw_data.numerical_cols)

    enc_data = EncodedDatasetWrapper.from_dataset(
        raw_data,
        col_trf,
    )
    return enc_data, col_trf


def load_compas(
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
    """
    ProPublica COMPAS with binary sensitive attribute (1 = African-American, 0 = Caucasian).

    Notes
    -----
    * ProPublica’s published preprocessing keeps only Black and White defendants,
      treating “race” as a two-class protected variable.:contentReference[oaicite:0]{index=0}
    * After filtering, we create an **integer** column ``race_binary`` and treat it
      as numerical so the encoded matrix is fully continuous.
    """
    COMPAS_URL = (
        "https://raw.githubusercontent.com/"
        "propublica/compas-analysis/master/compas-scores-two-years.csv"
    )
    BASE_PATH = find_root()
    TETRAD_METADATA_FPATH = BASE_PATH / "data/compas/metadata.json"

    SENSITIVE_NAME = "race"
    TARGET_NAME = "two_year_recid"

    df = pd.read_csv(COMPAS_URL, low_memory=False)
    df = apply_metadata_conversions(df, TETRAD_METADATA_FPATH)

    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df[SENSITIVE_NAME] = (df["race"] == "African-American").astype(int)

    numerical_cols = [
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
    ]
    categorical_cols = ["sex", "c_charge_degree", SENSITIVE_NAME]

    X = df[numerical_cols + categorical_cols]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    sensitive_train = X_train[SENSITIVE_NAME]
    sensitive_test = X_test[SENSITIVE_NAME]

    raw_data = DatasetWrapper(
        Xy_train=pd.concat([X_train, y_train], axis=1),
        Xy_test=pd.concat([X_test, y_test], axis=1),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="compas",
    )

    col_trf = get_column_transformer(
        raw_data.categorical_cols,
        raw_data.numerical_cols,
    )
    enc_data = EncodedDatasetWrapper.from_dataset(raw_data, col_trf)
    return enc_data, col_trf


def load_law_school(
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
    """
    Load Law School dataset directly from GitHub,
    split, wrap, and encode.
    """
    # download CSV
    LAW_SCHOOL_URL = (
        "https://raw.githubusercontent.com/"
        "damtharvey/law-school-dataset/main/law_dataset.csv"
    )
    df = pd.read_csv(LAW_SCHOOL_URL)
    df.rename(columns={"racetxt": "race"}, inplace=True)
    # define
    SENSITIVE_NAME = "race"
    TARGET_NAME = "pass_bar"
    numerical_cols = ["lsat", "ugpa"]
    categorical_cols = [SENSITIVE_NAME]

    # split
    X = df[numerical_cols + categorical_cols]
    y = df[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # wrap
    raw = DatasetWrapper(
        Xy_train=pd.concat([X_train, y_train], axis=1),
        Xy_test=pd.concat([X_test, y_test], axis=1),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=X_train[SENSITIVE_NAME],
        sensitive_test=X_test[SENSITIVE_NAME],
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="law_school",
    )

    # encode
    col_trf = get_column_transformer(categorical_cols, numerical_cols)
    enc_data = EncodedDatasetWrapper.from_dataset(raw, col_trf)
    return enc_data, col_trf


# # ------------------------------------------------------------------ #
# # 1. Bank-Marketing (UCI) — sensitive = “young” (< 25 yrs)​​:contentReference[oaicite:0]{index=0}
# # ------------------------------------------------------------------ #
# def load_bank_marketing(
#     test_size: float = 0.2,
#     random_state: int | None = 42,
# ) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
#     TARGET_NAME = "deposit"
#     SENSITIVE_NAME = "age_binary"

#     bank = fetch_bank_marketing(as_frame=True)
#     X = bank.data.copy()
#     y = bank.target.map({"no": 0, "yes": 1}).rename(TARGET_NAME)

#     # young (< 25) = 1, otherwise 0
#     X[SENSITIVE_NAME] = (X["age"] < 25).astype(int)

#     # basic column typing
#     categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
#     numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     if SENSITIVE_NAME not in numerical_cols:
#         numerical_cols.append(SENSITIVE_NAME)

#     # train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, stratify=y, random_state=random_state
#     )

#     sensitive_train = X_train[SENSITIVE_NAME]
#     sensitive_test = X_test[SENSITIVE_NAME]

#     raw = DatasetWrapper(
#         Xy_train=pd.concat([X_train, y_train], axis=1),
#         Xy_test=pd.concat([X_test, y_test], axis=1),
#         X_train=X_train,
#         X_test=X_test,
#         y_train=y_train,
#         y_test=y_test,
#         sensitive_train=sensitive_train,
#         sensitive_test=sensitive_test,
#         categorical_cols=categorical_cols,
#         numerical_cols=numerical_cols,
#         target_name=TARGET_NAME,
#         sensitive_name=SENSITIVE_NAME,
#         dataset_name="bank_marketing",
#     )

#     col_trf = get_column_transformer(raw.categorical_cols, raw.numerical_cols)
#     enc = EncodedDatasetWrapper.from_dataset(raw, col_trf)
#     return enc, col_trf


# # ------------------------------------------------------------------ #
# # 2. Diabetes-Hospital — sensitive = “race” (Afr-Am vs Caucasian)​​:contentReference[oaicite:1]{index=1}
# # ------------------------------------------------------------------ #
# def load_diabetes_hospital(
#     test_size: float = 0.2,
#     random_state: int | None = 42,
# ) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
#     TARGET_NAME = "readmit_binary"
#     SENSITIVE_NAME = "race_binary"

#     diab = fetch_diabetes_hospital(as_frame=True)
#     X = diab.data.copy()
#     y = diab.target.rename(TARGET_NAME)

#     # keep two largest groups → binarize
#     keep_races = ["African American", "Caucasian"]
#     X = X[X["race"].isin(keep_races)].copy()
#     y = y.loc[X.index]
#     X[SENSITIVE_NAME] = (X["race"] == "African American").astype(int)

#     categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
#     numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     if SENSITIVE_NAME not in numerical_cols:
#         numerical_cols.append(SENSITIVE_NAME)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, stratify=y, random_state=random_state
#     )
#     sensitive_train = X_train[SENSITIVE_NAME]
#     sensitive_test = X_test[SENSITIVE_NAME]

#     raw = DatasetWrapper(
#         Xy_train=pd.concat([X_train, y_train], axis=1),
#         Xy_test=pd.concat([X_test, y_test], axis=1),
#         X_train=X_train,
#         X_test=X_test,
#         y_train=y_train,
#         y_test=y_test,
#         sensitive_train=sensitive_train,
#         sensitive_test=sensitive_test,
#         categorical_cols=categorical_cols + [SENSITIVE_NAME],
#         numerical_cols=numerical_cols,
#         target_name=TARGET_NAME,
#         sensitive_name=SENSITIVE_NAME,
#         dataset_name="diabetes_hospital",
#     )

#     col_trf = get_column_transformer(raw.categorical_cols, raw.numerical_cols)
#     enc = EncodedDatasetWrapper.from_dataset(raw, col_trf)
#     return enc, col_trf


# ------------------------------------------------------------------ #
# 3. Credit-Card Default — sensitive = “SEX” (1 = male, 2 = female)​​:contentReference[oaicite:2]{index=2}
# ------------------------------------------------------------------ #
def load_credit_card(
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[EncodedDatasetWrapper, ColumnTransformer]:
    TARGET_NAME = "default"
    SENSITIVE_NAME = "sex_binary"

    cc = fetch_credit_card(as_frame=True)
    X = cc.data.copy()
    y = cc.target.rename(TARGET_NAME)

    # 1 = male, 2 = female → 1 if female
    X[SENSITIVE_NAME] = (X["SEX"] == 2).astype(int)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if SENSITIVE_NAME not in numerical_cols:
        numerical_cols.append(SENSITIVE_NAME)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    sensitive_train = X_train[SENSITIVE_NAME]
    sensitive_test = X_test[SENSITIVE_NAME]

    raw = DatasetWrapper(
        Xy_train=pd.concat([X_train, y_train], axis=1),
        Xy_test=pd.concat([X_test, y_test], axis=1),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="credit_card",
    )

    col_trf = get_column_transformer(raw.categorical_cols, raw.numerical_cols)
    enc = EncodedDatasetWrapper.from_dataset(raw, col_trf)
    return enc, col_trf
