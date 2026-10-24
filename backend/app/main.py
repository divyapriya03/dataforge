import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core import database as db
from app.api.endpoints import router as api_router
from app.services.preprocessor import DataPreprocessor

# Initialize SQLite tables on startup
db.init_db()

# Seed default student performance dataset if not already present
sample_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads", "student_performance.csv")
if os.path.exists(sample_csv) and not db.get_dataset("ds_student_01"):
    try:
        row_c, col_c, schema, _ = DataPreprocessor.inspect_csv(sample_csv)
        db.insert_dataset("ds_student_01", "student_performance.csv", sample_csv, row_c, col_c, schema)
    except Exception as e:
        print("Error seeding default dataset:", e)

app = FastAPI(
    title="DataForge API",
    description="Deep Learning-Based Tabular Synthetic Data Generation Platform API",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all REST endpoints
app.include_router(api_router)

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    deep_learning_engine: str

@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "online",
        "message": "DataForge Deep Learning API is running successfully",
        "version": "1.0.0",
        "deep_learning_engine": "VAE Neural Engine (Active)"
    }

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "online",
        "message": "DataForge Backend is operational and ready for requests",
        "version": "1.0.0",
        "deep_learning_engine": "VAE Neural Engine (Active)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
