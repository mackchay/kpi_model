import pandas as pd

def load_dataset(path: str):
    df = pd.read_csv(path)
    return df
