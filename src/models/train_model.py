import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

import sys
from pathlib import Path

from utils.mlflow_config import configure_mlflow


class MovieRecommenderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.item_lookup = pd.read_csv(context.artifacts["item_lookup"])
        self.sim_matrix = joblib.load(context.artifacts["sim_matrix"])
        
    def predict(self, model_input: pd.DataFrame):
        
        results = []
        
        for _, row in model_input.iterrows():
            query = str(row.get("title", ""))
            top_k = int(row.get("top_k", 5))
            
            matches = self.item_lookup[
                self.item_lookup["title"].str.contains(query, case=False, na=False)
            ]
            
            if len(matches) == 0:
                results.append([])
                continue
            
            idx = matches.index[0]
            
            sim_scores = list(enumerate(self.sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            top_indices = [i for i, s in sim_scores[1 : top_k + 1]]
            recommended = self.item_lookup.iloc[top_indices]["title"].tolist()
            
            results.append(recommended)
        return results
    
def load_items():
    df = pd.read_csv("data/processed/interactions.csv")
    return df

def build_item_similarity(items, max_features=5000):
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(items["title"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return tfidf, sim_matrix

def main():
    configure_mlflow()
    
    experiment_name = "movielens-content-based"
    mlflow.set_experiment(experiment_name)

    items = load_items()
    tfidf, sim_matrix = build_item_similarity(items)

    os.makedirs("models", exist_ok=True)
    joblib.dump(sim_matrix, "models/sim_matrix.joblib")
    items.to_csv("models/item_lookup.csv", index=False)
    
    with mlflow.start_run(run_name="v1") as run:
        mlflow.log_param("model_type", "content_based")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_metric("num_items", len(items))
        
        artifacts = {
            "sim_matrix": "models/sim_matrix.joblib",
            "item_lookup": "models/item_lookup.csv",
        }
        
        input_example = pd.DataFrame({
            "title": ["Toy Story", "Philadelphia Story"],
            "top_k": [5, 5],
        })
        
        output_example = [['Philadelphia (1993)',
                'Pinocchio (1940)',
                'Fantasia (1940)',
                'Rebecca (1940)',
                'Toy Story (1995)'],
                ["Pyromaniac's Love Story, A (1995)",
                'Story of Xinghua, The (1993)',
                'Philadelphia Story, The (1940)',
                'NeverEnding Story III, The (1994)',
                'FairyTale: A True Story (1997)']]
        
        signature = infer_signature(input_example, output_example)
        
        mlflow.pyfunc.log_model(
            name="model",
            python_model=MovieRecommenderModel(),
            artifacts=artifacts,
            input_example=input_example,
            signature=signature
        )
        
        run_id = run.info.run_id
        print(f"Logged model to run_id={run_id}")
        
if __name__ == "__main__":
    main()
        
        

            
               
        
        