import os
import pandas as pd

def load_raw_movielens():
    item_path = os.path.join("data", "raw", "ml-100k", "u.item")
    items = pd.read_csv(
    item_path,
    sep="|",
    header=None,
    encoding="latin-1",
    usecols=[0, 1],
    names=["item_id", "title"]  
    )
    
    return items

def preprocess(df):
    df = df.dropna(subset=["title"])
    return df

def save_processed(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/interactions.csv", index=False)
    
if __name__ == "__main__":
    df = load_raw_movielens()
    df = preprocess(df)
    save_processed(df)
    print("Saved processed data to data/processed/interactions.csv")

