import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core import config
from app.core import database as db
from app.api.endpoints import router as api_router
from app.services.preprocessor import DataPreprocessor

log = logging.getLogger("dataforge")

# Initialize SQLite tables on startup
db.init_db()

# Seed default student performance dataset if not already present
sample_csv = os.path.join(config.get_dir("uploads"), "student_performance.csv")
if os.path.exists(sample_csv) and not db.get_dataset("ds_student_01"):
    try:
        row_c, col_c, schema, _ = DataPreprocessor.inspect_csv(sample_csv)
        db.insert_dataset("ds_student_01", "student_performance.csv", config.to_stored_path(sample_csv), row_c, col_c, schema)
    except Exception as e:
        log.warning("Could not seed the default dataset: %s", e)

app = FastAPI(
    title="DataForge API",
    description="Deep Learning-Based Tabular Synthetic Data Generation Platform API",
    version="1.0.0"
)

# CORS: only the configured browser origins (default: the Vite dev/preview servers) may call the API.
# No cookies or credentials are used by the application, so credentialed requests are not allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Mount all REST endpoints
app.include_router(api_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Details go to the server log only; clients get a generic message.
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})

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
