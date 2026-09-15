import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.db import Base, engine
import app.models  # noqa: F401 - ensure models are registered
from app.routers.chat import router as chat_router
from app.routers.plans import router as plans_router
from app.routers.runs import router as runs_router

# Make lifecycle/observability logs visible under plain `uvicorn app.main:app`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retail_workbench")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured; application startup complete")
    yield


app = FastAPI(title="retail-data-workbench", lifespan=lifespan)

# Allow the Vite dev server (http://localhost:5173) to call the backend
# directly during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    """Attach a request id + timing to every request (observability baseline).

    Honours an incoming X-Request-Id (e.g. set by a proxy) so requests can be
    traced across systems; otherwise one is generated. The id is echoed in the
    response and every log line for the request, and unhandled exceptions are
    logged with it before propagating.
    """
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "[%s] %s %s CRASHED after %.1f ms",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Long-running endpoints (profile/clean on big files) are the ones worth
    # watching; everything else stays at INFO for the request trail.
    logger.info(
        "[%s] %s %s -> %d (%.1f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-Id"] = request_id
    return response


app.include_router(runs_router)
app.include_router(plans_router)
app.include_router(chat_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
