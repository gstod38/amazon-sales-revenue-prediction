import pandas as pd

LEAKAGE_COLUMNS = [
    "discounted_price",
    "quantity_sold"
]

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing on the Amazon sales dataset.
    - Parse dates
    - Drop leakage columns
    - Handle missing values
    - Ensure correct data types
    """
    df = df.copy()

    # Parse order date
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Drop leakage columns
    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    # Handle missing values (light-touch)
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    return df
