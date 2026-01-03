import pandas as pd
import mlflow
import sys
from pathlib import Path
from utils.mlflow_config import configure_mlflow

model_name = "movie-recommender"
model_version = 1

def main():
   configure_mlflow()
   
   model_uri = f"models:/{model_name}/{model_version}"
   
   model = mlflow.pyfunc.load_model(model_uri)
   
   print(f"Loaded model from {model_uri}")
   
   input_df = pd.DataFrame(
        [
            {"title": "Philadelphia Story", "top_k": 5},
            {"title": "Toy Story", "top_k": 5},
        ]
    )
   
   preds = model.predict(input_df)
   
   for query, recs in zip(input_df["title"], preds):
        print(f"Recommendations for '{query}':")
        for r in recs:
            print("  -", r)
            
if __name__ == "__main__":
    main()