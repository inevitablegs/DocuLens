"""
DocuLens AI — FastAPI application entry point
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from backend.database.db import init_db, async_session
from backend.database.models import Document
from backend.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup and recover stale processing jobs."""
    await init_db()
    # Recover any stale 'processing' documents from server restarts
    try:
        async with async_session() as session:
            result = await session.execute(
                select(Document).where(Document.status == "processing")
            )
            stale_docs = result.scalars().all()
            for d in stale_docs:
                d.status = "failed"
            if stale_docs:
                await session.commit()
                logger.info(f"Recovered {len(stale_docs)} stale processing documents on startup.")
    except Exception as e:
        logger.warning(f"Failed to recover stale documents on startup: {e}")
    yield


app = FastAPI(
    title="DocuLens AI",
    description="Intelligent Document Processing & Insight Generator",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "DocuLens AI",
        "version": "1.0.0",
        "status": "running",
    }
