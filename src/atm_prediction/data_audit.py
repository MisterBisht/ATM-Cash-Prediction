
import pandas as pd

TRAIN = "data/raw/train.csv"
LATER = "data/raw/later.csv"
HOLDOUT = "data/raw/holdout_features.csv"

train = pd.read_csv(TRAIN)
later = pd.read_csv(LATER)
holdout = pd.read_csv(HOLDOUT)

target = "withdrawal_thousand_rs"

print("\n========== DATASET SHAPES ==========")
print("Train:", train.shape)
print("Later:", later.shape)
print("Holdout:", holdout.shape)

print("\n========== 1. POSSIBLE LEAKAGE ==========")
extra_cols = set(train.columns) - set(holdout.columns)
print("Train-only columns:", extra_cols)

for col in extra_cols:
    if pd.api.types.is_numeric_dtype(train[col]):
        print(
            col,
            "correlation with target:",
            train[col].corr(train[target])
        )

print("\n========== 2. UNIT MIXUP ==========")
numeric_cols = train.select_dtypes(include="number").columns

for col in numeric_cols:
    if col == target:
        continue
    print(f"\n{col}")
    print("Train median by area:")
    print(train.groupby("area")[col].median())
    if col in later.columns:
        print("Later median by area:")
        print(later.groupby("area")[col].median())

print("\n========== 3. SENTINEL VALUES ==========")
sentinels = [-1, -9, -99, -999, -9999, 99, 999, 9999]

for col in numeric_cols:
    counts = train[col].value_counts()
    found = {v: int(counts[v]) for v in sentinels if v in counts.index}
    if found:
        print(col, found)

print("\n========== 4. DUPLICATE ROWS ==========")
print("Exact duplicate rows:", train.duplicated().sum())

print("\n========== 5. STUCK SENSOR ==========")
sorted_train = train.sort_values("date")

for col in numeric_cols:
    if col == target:
        continue

    values = sorted_train[col].reset_index(drop=True)
    groups = values.ne(values.shift()).cumsum()
    runs = (
        sorted_train.assign(_group=groups)
        .groupby("_group")
        .agg(
            value=(col, "first"),
            count=(col, "size"),
            start=("date", "first"),
            end=("date", "last")
        )
    )
    long_runs = runs[runs["count"] >= 10]

    if not long_runs.empty:
        print(f"\n{col}:")
        print(long_runs.to_string(index=False))

print("\n========== 6. TARGET OUTLIERS ==========")
print(train[target].describe())
median = train[target].median()
print("Values above 5x median:")
print(train.loc[train[target] > 5 * median,
                ["record_id", target]].to_string(index=False))

print("\n========== 7. CATEGORY VARIANTS ==========")
for col in ["atm_type", "area"]:
    print(f"\n{col} - TRAIN:")
    print(train[col].value_counts(dropna=False).to_string())
    print(f"\n{col} - LATER:")
    print(later[col].value_counts(dropna=False).to_string())

print("\n========== AUDIT COMPLETE ==========")

print("\n========== SENTINEL VERIFICATION ==========")

for col in ["nearby_shops", "footfall_index"]:
    print(f"\n{col}")
    print("Train rows having value 99:")
    print(train.loc[train[col] == 99,
                    ["record_id", "date", "area", col]].to_string(index=False))

    print("Train range:", train[col].min(), train[col].max())
    print("Later range:", later[col].min(), later[col].max())


print("\n========== TARGET TYPO VERIFICATION ==========")

outliers = train[train[target] > 5 * train[target].median()].copy()

outliers["value_divided_by_10"] = outliers[target] / 10

print(outliers[
    ["record_id", target, "value_divided_by_10"]
].to_string(index=False))

print("\nNormal target distribution:")
print(train.loc[train[target] < 2000, target].describe())


print("\n========== LEAKAGE VERIFICATION ==========")

print("Train columns:", train.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())
print("Train-only columns excluding target:",
      set(train.columns) - set(holdout.columns) - {target})
