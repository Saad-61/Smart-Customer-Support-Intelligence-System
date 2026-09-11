"""
Riot Games Smart Customer Support Intelligence System
FastAPI REST Inference Service (Module 11)

This service serves real-time multi-task machine learning predictions for player support tickets:
  1. Category Classification (Platt-scaled LinearSVC calibrated probabilities)
  2. Priority Prediction (XGBoost classifier on GPU)
  3. Dense Semantic Ticket Retrieval (SentenceTransformer 'all-MiniLM-L6-v2' on CUDA)
  4. Model Explainability (Linear hyperplane feature attribution & contributions)
  5. Out-of-Distribution (OOD) Guardrail (Flags uncertain predictions below 50% confidence)

Architecture:
  - All model artifacts and transformer weights are loaded ONCE at startup via FastAPI lifespan.
  - No retraining or model disk reads occur during request handling.
  - Zero disk I/O per inference request yields ultra-low end-to-end response latencies.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register PriorityXGBClassifier alias for unpickling
import src.train
sys.modules.setdefault("__main__", sys.modules["src.train"])
if not hasattr(sys.modules["__main__"], "PriorityXGBClassifier"):
    setattr(sys.modules["__main__"], "PriorityXGBClassifier", src.train.PriorityXGBClassifier)

from src.evaluate import explain_category_prediction
from src.similarity import find_similar_tickets, load_retrieval_index
from src.train import load_category_model, load_priority_model

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("riot_support_api")


# ===========================================================================
# 1. PYDANTIC DATA SCHEMAS
# ===========================================================================

class TicketRequest(BaseModel):
    """Incoming player support complaint request payload."""
    ticket_text: str = Field(
        ...,
        description="Raw incoming player complaint text",
        examples=["I was charged twice for the same RP bundle"],
    )
    product: str = Field(
        default="League of Legends",
        description="Riot Games product/game title",
        examples=["League of Legends"],
    )
    previous_tickets: int = Field(
        default=0,
        description="Customer historical ticket count",
        ge=0,
        examples=[0],
    )

    @field_validator("ticket_text")
    @classmethod
    def validate_ticket_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("ticket_text must not be empty or contain only whitespace.")
        return stripped


class SimilarTicket(BaseModel):
    """Schema for a semantically similar historical support ticket."""
    ticket_id: str = Field(..., description="Unique identifier of historical ticket")
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    preview: str = Field(..., description="First 80-100 characters of ticket text")
    category: str = Field(default="", description="Assigned historical category")
    priority: str = Field(default="", description="Assigned historical priority")
    product: str = Field(default="", description="Associated Riot Games product")


class ExplanationFeature(BaseModel):
    """Salient token or bigram influencing the category classification decision."""
    feature: str = Field(..., description="Token, bigram, or metadata feature name")
    weight: float = Field(..., description="Model class coefficient weight")
    contribution: float = Field(..., description="Local feature contribution (x_j * w_j)")


class PredictionResponse(BaseModel):
    """Comprehensive multi-task triage response payload."""
    category: str = Field(..., description="Predicted issue category")
    priority: str = Field(..., description="Predicted urgency priority (LOW, MEDIUM, HIGH)")
    category_confidence: float = Field(..., description="Platt-scaled calibrated probability")
    priority_confidence: float = Field(..., description="Predicted priority class probability")
    calibrated_note: str = Field(
        default="Confidence is Platt-scaled (sigmoid calibration). Not a raw model score.",
        description="Explanation of calibration methodology",
    )
    similar_tickets: list[SimilarTicket] = Field(
        ..., description="Top semantically similar historical support tickets"
    )
    explanation: list[ExplanationFeature] = Field(
        ..., description="Top feature attributions explaining the category prediction"
    )
    uncertain: bool = Field(
        ..., description="True if category confidence < OOD threshold (0.50), indicating ambiguous or out-of-distribution input"
    )
    processing_time_ms: float = Field(
        ..., description="Total server-side processing latency in milliseconds"
    )


class HealthResponse(BaseModel):
    """System health check and loaded artifact status."""
    status: str
    models_loaded: bool
    device: str
    gpu_name: str | None
    loaded_models: dict[str, str]
    total_indexed_tickets: int


class SimilarRequest(BaseModel):
    """Standalone semantic retrieval request."""
    query_text: str = Field(..., description="Query complaint text to search", examples=["Cannot connect to chat server"])
    top_k: int = Field(default=5, ge=1, le=50, description="Number of similar tickets to retrieve")

    @field_validator("query_text")
    @classmethod
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query_text must not be empty or contain only whitespace.")
        return stripped


class SimilarResponse(BaseModel):
    """Standalone semantic retrieval response."""
    query: str
    top_k: int
    results: list[SimilarTicket]
    processing_time_ms: float


class ExplainRequest(BaseModel):
    """Standalone model explainability request."""
    ticket_text: str = Field(..., description="Ticket text to explain", examples=["My account was permanently suspended for third party software"])
    product: str = Field(default="League of Legends", description="Product title")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of top features to return")

    @field_validator("ticket_text")
    @classmethod
    def validate_ticket(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("ticket_text must not be empty or contain only whitespace.")
        return stripped


class ExplainResponse(BaseModel):
    """Standalone model explainability response."""
    ticket_text: str
    predicted_category: str
    confidence: float
    top_features: list[ExplanationFeature]
    processing_time_ms: float


# ===========================================================================
# 2. APPLICATION LIFESPAN & STATE INITIALIZATION
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    Loads all trained model artifacts, vectorizers, dense retrieval index,
    and neural sentence transformer onto GPU/CPU once at startup.
    """
    logger.info("Initializing Riot Games Smart Customer Support Intelligence Service...")

    # Detect hardware accelerator
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA GPU detected: %s (device='%s')", gpu_name, device)
    else:
        device = "cpu"
        gpu_name = None
        logger.info("CUDA not available. Running inference on CPU.")

    models_dir = REPO_ROOT / "models"

    # 1. Load Category Classification Pipeline (Platt-Calibrated LinearSVC)
    category_path = models_dir / "category_model.joblib"
    if not category_path.exists():
        logger.error("Category model not found at %s", category_path)
        app.state.category_model = None
    else:
        logger.info("Loading Category Classification model from %s...", category_path.name)
        app.state.category_model = load_category_model(category_path)
        logger.info("Category model loaded successfully.")

    # 2. Load Priority Prediction Pipeline (GPU XGBoost)
    priority_path = models_dir / "priority_model.joblib"
    if not priority_path.exists():
        logger.error("Priority model not found at %s", priority_path)
        app.state.priority_model = None
    else:
        logger.info("Loading Priority Prediction model from %s...", priority_path.name)
        app.state.priority_model = load_priority_model(priority_path)
        logger.info("Priority model loaded successfully.")

    # 3. Load Semantic Retrieval Index (Dense Embeddings)
    index_path = models_dir / "retrieval_index.joblib"
    if not index_path.exists():
        logger.error("Retrieval index not found at %s", index_path)
        app.state.retrieval_index = None
    else:
        logger.info("Loading dense semantic retrieval index from %s...", index_path.name)
        app.state.retrieval_index = load_retrieval_index(index_path)
        total_tickets = app.state.retrieval_index.get("total_records", 0)
        logger.info("Retrieval index loaded successfully (%d indexed tickets).", total_tickets)

    # 4. Pre-warm Sentence Transformer on GPU
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"
        logger.info("Pre-warming SentenceTransformer('%s') on %s...", model_name, device)
        app.state.sentence_model = SentenceTransformer(model_name, device=device)
        logger.info("SentenceTransformer initialized and ready on %s.", device)
    except Exception as exc:
        logger.warning("Could not pre-load SentenceTransformer: %s", exc)
        app.state.sentence_model = None

    app.state.device = device
    app.state.gpu_name = gpu_name
    app.state.startup_time = datetime.now().isoformat()
    app.state.models_loaded = (
        app.state.category_model is not None
        and app.state.priority_model is not None
        and app.state.retrieval_index is not None
    )

    logger.info("Service startup complete. All models loaded: %s", app.state.models_loaded)
    yield

    # Teardown logic
    logger.info("Shutting down Riot Support API service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===========================================================================
# 3. FASTAPI APPLICATION DEFINITION
# ===========================================================================

app = FastAPI(
    title="Riot Games Support Intelligence System API",
    description=(
        "Production-grade REST inference service for intelligent triage of player support tickets. "
        "Provides Platt-calibrated category classification, XGBoost priority prediction, "
        "CUDA-accelerated dense semantic retrieval, linear feature attribution, and OOD uncertainty detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend dashboard or cross-origin consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# 4. HELPER FUNCTIONS
# ===========================================================================

def verify_models_ready(app_instance: FastAPI) -> None:
    """Ensure all core machine learning models are loaded into application state."""
    if not getattr(app_instance.state, "models_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Models not loaded or failed to initialize. "
                "Ensure category_model.joblib, priority_model.joblib, and retrieval_index.joblib exist in models/."
            ),
        )


# ===========================================================================
# 5. ENDPOINTS
# ===========================================================================

@app.get(
    "/",
    tags=["General"],
    summary="API Root Information",
    description="Returns service metadata, documentation links, and operational status.",
)
def root():
    return {
        "service": "Riot Games Smart Customer Support Intelligence System API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "predict": "POST /predict",
            "similar": "POST /similar",
            "explain": "POST /explain",
            "health": "GET /health",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service & Model Health Check",
    description="Inspects memory status, device accelerator (CUDA GPU / CPU), and artifact load readiness.",
)
def health(request: Request):
    is_ready = getattr(request.app.state, "models_loaded", False)
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning pipelines are not ready.",
        )

    retrieval_idx = getattr(request.app.state, "retrieval_index", {}) or {}
    total_tickets = retrieval_idx.get("total_records", 0)

    return HealthResponse(
        status="ok",
        models_loaded=True,
        device=getattr(request.app.state, "device", "cpu"),
        gpu_name=getattr(request.app.state, "gpu_name", None),
        loaded_models={
            "category_model": "LinearSVC + 3-Fold Platt Sigmoid Calibration",
            "priority_model": "TruncatedSVD(50) + OneHotEncoder + XGBoost GPU",
            "retrieval_index": "SentenceTransformer('all-MiniLM-L6-v2') 384-d Cosine Index",
        },
        total_indexed_tickets=total_tickets,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Multi-Task Support Ticket Triage",
    description=(
        "End-to-end triage prediction: predicts issue category with Platt-calibrated probability, "
        "flags out-of-distribution queries with uncertain=True, predicts ticket priority, "
        "extracts top explanatory keywords/bigrams, and retrieves semantically similar historical tickets."
    ),
)
def predict(request: Request, payload: TicketRequest):
    verify_models_ready(request.app)
    start_time = time.perf_counter()

    category_model = request.app.state.category_model
    priority_model = request.app.state.priority_model
    retrieval_index = request.app.state.retrieval_index
    sentence_model = getattr(request.app.state, "sentence_model", None)
    device = getattr(request.app.state, "device", None)

    # 1. Assemble structured feature row
    now = datetime.now()
    feature_df = pd.DataFrame([{
        "ticket_text": payload.ticket_text,
        "product": payload.product,
        "previous_tickets": payload.previous_tickets,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "month": now.month,
    }])

    # 2. Predict Category + Platt Calibrated Probability
    category_pred = str(category_model.predict(feature_df)[0])
    clf = category_model.named_steps["clf"]
    classes = list(clf.classes_)
    pred_idx = classes.index(category_pred)

    if hasattr(category_model, "predict_proba"):
        cat_probas = category_model.predict_proba(feature_df)[0]
        category_confidence = float(cat_probas[pred_idx])
        max_confidence = float(np.max(cat_probas))
    else:
        category_confidence = 1.0
        max_confidence = 1.0

    # 3. Out-of-Distribution (OOD) Guardrail
    # If maximum calibrated confidence is below 0.50, flag as uncertain
    uncertain = bool(max_confidence < 0.50)

    # 4. Predict Priority + Confidence
    priority_pred = str(priority_model.predict(feature_df)[0])
    pri_clf = priority_model.named_steps["clf"]
    pri_classes = list(pri_clf.classes_)
    pri_idx = pri_classes.index(priority_pred)

    if hasattr(priority_model, "predict_proba"):
        pri_probas = priority_model.predict_proba(feature_df)[0]
        priority_confidence = float(pri_probas[pri_idx])
    else:
        priority_confidence = 1.0

    # 5. Extract Feature Attribution Explanation
    try:
        explanation_data = explain_category_prediction(
            text=payload.ticket_text,
            pipeline=category_model,
            top_n=5,
            product=payload.product,
            customer_metadata={"previous_tickets": payload.previous_tickets},
        )
        explanation_features = [
            ExplanationFeature(
                feature=feat["feature"],
                weight=round(float(feat["weight"]), 4),
                contribution=round(float(feat.get("contribution", 0.0)), 4),
            )
            for feat in explanation_data.get("top_features", [])
        ]
    except Exception as exc:
        logger.warning("Explainability extraction failed: %s", exc)
        explanation_features = []

    # 6. Retrieve Similar Historical Tickets
    try:
        raw_similar = find_similar_tickets(
            query_text=payload.ticket_text,
            index=retrieval_index,
            top_k=3,
            model=sentence_model,
            device=device,
        )
        similar_tickets = [
            SimilarTicket(
                ticket_id=item["ticket_id"],
                similarity=round(float(item["similarity"]), 4),
                preview=item.get("preview", item.get("text", "")[:80]),
                category=item.get("category", ""),
                priority=item.get("priority", ""),
                product=item.get("product", ""),
            )
            for item in raw_similar
        ]
    except Exception as exc:
        logger.warning("Similar ticket retrieval failed: %s", exc)
        similar_tickets = []

    # 7. Compute Latency and Assemble Response
    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return PredictionResponse(
        category=category_pred,
        priority=priority_pred,
        category_confidence=round(category_confidence, 4),
        priority_confidence=round(priority_confidence, 4),
        calibrated_note="Confidence is Platt-scaled (sigmoid calibration). Not a raw model score.",
        similar_tickets=similar_tickets,
        explanation=explanation_features,
        uncertain=uncertain,
        processing_time_ms=processing_time_ms,
    )


@app.post(
    "/similar",
    response_model=SimilarResponse,
    tags=["Inference"],
    summary="Semantic Similar Ticket Retrieval",
    description="Retrieve top-k semantically similar historical support tickets via dense neural embeddings.",
)
def get_similar_tickets(request: Request, payload: SimilarRequest):
    verify_models_ready(request.app)
    start_time = time.perf_counter()

    retrieval_index = request.app.state.retrieval_index
    sentence_model = getattr(request.app.state, "sentence_model", None)
    device = getattr(request.app.state, "device", None)

    results = find_similar_tickets(
        query_text=payload.query_text,
        index=retrieval_index,
        top_k=payload.top_k,
        model=sentence_model,
        device=device,
    )

    tickets = [
        SimilarTicket(
            ticket_id=item["ticket_id"],
            similarity=round(float(item["similarity"]), 4),
            preview=item.get("preview", item.get("text", "")[:80]),
            category=item.get("category", ""),
            priority=item.get("priority", ""),
            product=item.get("product", ""),
        )
        for item in results
    ]

    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return SimilarResponse(
        query=payload.query_text,
        top_k=payload.top_k,
        results=tickets,
        processing_time_ms=processing_time_ms,
    )


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["Explainability"],
    summary="Linear Hyperplane Feature Attribution",
    description="Computes local feature contributions for an incoming complaint using Platt-scaled linear coefficients.",
)
def explain_ticket(request: Request, payload: ExplainRequest):
    verify_models_ready(request.app)
    start_time = time.perf_counter()

    category_model = request.app.state.category_model

    explanation_data = explain_category_prediction(
        text=payload.ticket_text,
        pipeline=category_model,
        top_n=payload.top_n,
        product=payload.product,
    )

    features = [
        ExplanationFeature(
            feature=feat["feature"],
            weight=round(float(feat["weight"]), 4),
            contribution=round(float(feat.get("contribution", 0.0)), 4),
        )
        for feat in explanation_data.get("top_features", [])
    ]

    processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return ExplainResponse(
        ticket_text=payload.ticket_text,
        predicted_category=explanation_data["predicted_category"],
        confidence=round(float(explanation_data["confidence"]), 4),
        top_features=features,
        processing_time_ms=processing_time_ms,
    )


# ===========================================================================
# 6. STANDALONE CLI TEST RUNNER
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("  Starting Riot Games Support Intelligence API Service (Development)")
    print("  Interactive Swagger UI Documentation: http://127.0.0.1:8000/docs")
    print("=" * 80 + "\n")

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
