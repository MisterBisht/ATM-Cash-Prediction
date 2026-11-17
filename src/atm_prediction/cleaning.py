
import pandas as pd


TARGET_TYPO_IDS = {
    "R00168",
    "R00363",
    "R00561",
    "R00595",
    "R00222",
    "R00041",
    "R00723",
    "R00407"
}


def normalize_atm_type(value):
    value = str(value).strip().lower()

    mapping = {
        "standalone": "standalone",
        "sta.": "standalone",

        "mall": "mall",
        "mal.": "mall",

        "bank_branch": "bank_branch",
        "bank branch": "bank_branch",
        "ban.": "bank_branch"
    }

    return mapping.get(value, value)


def clean_train(df):
    df = df.copy()

    # 1. Correct distance unit for Market area
    market_mask = df["area"].str.strip().str.lower() == "market"

    df.loc[market_mask, "distance_to_market_km"] = (
        df.loc[market_mask, "distance_to_market_km"] / 1000
    )

    # 2. Correct the eight target typos
    typo_mask = df["record_id"].isin(TARGET_TYPO_IDS)

    found_ids = set(df.loc[typo_mask, "record_id"])

    if found_ids != TARGET_TYPO_IDS:
        raise ValueError("Some target typo records are missing!")

    df.loc[typo_mask, "withdrawal_thousand_rs"] = (
        df.loc[typo_mask, "withdrawal_thousand_rs"] / 10
    )

    # 3. Standardize ATM type categories
    df["atm_type"] = df["atm_type"].apply(normalize_atm_type)

    return df


def clean_later(df):
    df = df.copy()

    # Later dataset is already clean.
    # Only standardize category labels for consistency.
    df["atm_type"] = df["atm_type"].apply(normalize_atm_type)

    return df
