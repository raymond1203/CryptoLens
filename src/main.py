import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.config import settings
from src.db.qdrant import ensure_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
    app.state.openai = AsyncOpenAI(api_key=settings.openai_api_key)

    await ensure_collection(app.state.qdrant)

    yield

    # Shutdown
    await app.state.qdrant.close()
    await app.state.redis.aclose()
    await app.state.anthropic.close()
    await app.state.openai.close()


app = FastAPI(title="CryptoLens", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    checks = {}

    try:
        await app.state.qdrant.get_collections()
        checks["qdrant"] = "ok"
    except Exception:
        logger.exception("Qdrant health check failed")
        checks["qdrant"] = "unavailable"

    try:
        await app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception:
        logger.exception("Redis health check failed")
        checks["redis"] = "unavailable"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
