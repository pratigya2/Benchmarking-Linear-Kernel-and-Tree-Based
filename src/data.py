import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, TARGET, ID_COLS, LEAK_COLS, SEED, TEST_SIZE


def load_raw():
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=ID_COLS + LEAK_COLS)


def split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=SEED, stratify=y)
