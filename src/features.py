import pandas as pd

CATEGORICAL_COLS = [
    "product_category",
    "customer_region",
    "payment_method"
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features from preprocessed Amazon sales data.
    """
    df = df.copy()

    # --- Time-based features ---
    df["order_month"] = df["order_date"].dt.month
    df["order_year"] = df["order_date"].dt.year

    # --- Drop raw date (signal already extracted) ---
    df = df.drop(columns=["order_date"], errors="ignore")

    # --- Drop non-predictive identifiers ---
    df = df.drop(columns=["order_id", "product_id"], errors="ignore")

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df

