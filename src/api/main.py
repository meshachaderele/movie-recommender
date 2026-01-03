from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import sys
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.mlflow_config import configure_mlflow


model_name = "movie-recommender"
model_version = 1

class RecommendRequest(BaseModel):
    title: str = "Toy Story"
    top_k: int = 5
    
class RecommendResponse(BaseModel):
    title: str
    recommendations: list[str]
    
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model on Startup
    configure_mlflow()
    model_uri = f"models:/{model_name}/{model_version}"
    app.state.model = mlflow.pyfunc.load_model(model_uri)
    print(f"Loaded model from {model_uri}")

    yield
    
app = FastAPI(
    title="Movie Recommender API",
    version="1.0.0",
    description="Content based movie recommender served from MLflow Model Registry.",
    lifespan=lifespan,
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    model = app.state.model
    
    input_df = pd.DataFrame(
        [
            {"title": req.title, "top_k": req.top_k}
        ]
    )
    preds = model.predict(input_df)
    recommendations = preds[0] if len(preds) > 0 else []
    
    return RecommendResponse(
        title=req.title,
        recommendations=recommendations,
    )