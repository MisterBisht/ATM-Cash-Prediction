
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


NUMERIC_FEATURES = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km"
]

CATEGORICAL_FEATURES = [
    "atm_type",
    "area"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data(df, test_size=0.2):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]
    valid_df = df.iloc[split_index:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["withdrawal_thousand_rs"]

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["withdrawal_thousand_rs"]

    return X_train, X_valid, y_train, y_valid


def build_pipeline(model_params, random_state=42):

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             CATEGORICAL_FEATURES)
        ]
    )

    model = RandomForestRegressor(
        **model_params,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
